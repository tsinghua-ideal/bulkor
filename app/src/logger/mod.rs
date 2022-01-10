use std::fmt::format;
use std::fs::{remove_file, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use log::error;
use once_cell::sync::OnceCell;
use simplelog::*;

/// Output information, not for functional logging (e.g., logging queries).
static LOGGER: OnceCell<()> = OnceCell::new();

pub(crate) fn initialize_loggers<P: Into<PathBuf>>(file_path: P, level: LogLevel) {
    fn _initializer(file_path: PathBuf, level: LogLevel) {
        let log_level = level.into();
        log::info!("path for file logger: {}", file_path.display());
        let file_logger: Box<dyn SharedLogger> = WriteLogger::new(
            log_level,
            Config::default(),
            File::create(file_path).expect("not able to create log file"),
        );
        let mut combined = vec![file_logger];
        if let Some(term_logger) =
            TermLogger::new(log_level, Config::default(), TerminalMode::Mixed)
        {
            let logger: Box<dyn SharedLogger> = term_logger;
            combined.push(logger);
        }
        CombinedLogger::init(combined).unwrap();
    }

    LOGGER.get_or_init(move || _initializer(file_path.into(), level));
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum LogLevel {
    Error,
    Warn,
    Debug,
    Trace,
    Info,
}

impl LogLevel {
    pub fn is_debug_or_lower(self) -> bool {
        use LogLevel::*;
        match self {
            Debug | Trace => true,
            _ => false,
        }
    }
}

impl Into<LevelFilter> for LogLevel {
    fn into(self) -> LevelFilter {
        match self {
            LogLevel::Error => LevelFilter::Error,
            LogLevel::Warn => LevelFilter::Warn,
            LogLevel::Debug => LevelFilter::Debug,
            LogLevel::Trace => LevelFilter::Trace,
            _ => LevelFilter::Info,
        }
    }
}


/// This is for logging queries.
#[no_mangle]
pub extern "C" fn log_query(
    id: u64,
    query: *const u8,
    query_len: usize,
) {
    let query = unsafe {
        core::slice::from_raw_parts(query, query_len)
    };
    let path_str = format!("log_{:?}", id);
    //think about hold the log somewhere, avoid opening it every time
    let mut log = OpenOptions::new().create(true).append(true).read(true).open(&path_str).unwrap();
    log.write_all(query).unwrap();
}
