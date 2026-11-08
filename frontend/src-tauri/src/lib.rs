use std::net::TcpStream;
use std::path::Path;
use std::process::{Child, Command};
use std::sync::Mutex;
use tauri::{Manager, RunEvent};

#[cfg(windows)]
use std::os::windows::io::AsRawHandle;
#[cfg(windows)]
use std::os::windows::process::CommandExt;

/// El `Job` debe vivir mientras viva la app: al cerrarse su manejador (porque la app termina, también
/// si se cuelga o la matan), Windows mata todo el árbol de procesos del backend, motores de IA incluidos.
struct Backend {
    child: Mutex<Option<Child>>,
    #[cfg(windows)]
    _job: Option<win32job::Job>,
}

/// Mete el proceso en un Job de Windows con "matar al cerrar": así el backend nunca queda huérfano.
#[cfg(windows)]
fn guard_with_job(child: &Child) -> Option<win32job::Job> {
    let job = win32job::Job::create().ok()?;
    let mut info = job.query_extended_limit_info().ok()?;
    info.limit_kill_on_job_close();
    job.set_extended_limit_info(&mut info).ok()?;
    job.assign_process(child.as_raw_handle() as isize).ok()?;
    Some(job)
}

/// Elige cómo arrancar el backend, de más a menos independiente del equipo:
/// 1. App instalada: `nova-backend.exe` empaquetado junto a los recursos (no necesita Python).
/// 2. Desarrollo: el Python del entorno del backend (`backend/.venv`), sin `uv run`.
/// 3. Último recurso: `python -m uv run python -m nova` (necesita Python y uv en el sistema).
fn backend_command(app: &tauri::AppHandle) -> Command {
    if let Ok(res) = app.path().resource_dir() {
        let dir = res.join("backend");
        let exe = dir.join("nova-backend.exe");
        if exe.is_file() {
            let mut c = Command::new(exe);
            c.current_dir(dir);
            return c;
        }
    }
    let backend = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../backend");
    let venv_python = backend.join(".venv/Scripts/python.exe");
    if venv_python.is_file() {
        let mut c = Command::new(venv_python);
        c.args(["-m", "nova"]).current_dir(backend);
        return c;
    }
    let mut c = Command::new("python");
    c.args(["-m", "uv", "run", "python", "-m", "nova"]).current_dir(backend);
    c
}

/// Arranca el backend. Si ya hay uno escuchando en el puerto 8765, no lanza otro.
fn spawn_backend(app: &tauri::AppHandle) -> Backend {
    if TcpStream::connect("127.0.0.1:8765").is_ok() {
        return Backend {
            child: Mutex::new(None),
            #[cfg(windows)]
            _job: None,
        };
    }
    let mut cmd = backend_command(app);
    #[cfg(windows)]
    cmd.creation_flags(0x08000000); // CREATE_NO_WINDOW
    let child = cmd.spawn().ok();
    #[cfg(windows)]
    let job = child.as_ref().and_then(guard_with_job);
    Backend {
        child: Mutex::new(child),
        #[cfg(windows)]
        _job: job,
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            let backend = spawn_backend(app.handle());
            app.manage(backend);
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building tauri application");

    app.run(|handle, event| {
        if let RunEvent::Exit = event {
            if let Some(mut child) = handle.state::<Backend>().child.lock().unwrap().take() {
                // /T cierra también los procesos hijos (los de los motores de IA)
                let _ = Command::new("taskkill")
                    .args(["/PID", &child.id().to_string(), "/T", "/F"])
                    .output();
                let _ = child.wait();
            }
        }
    });
}
