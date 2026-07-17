//! Thin command-line presentation for the RangeForge engine.

use rf_engine::action::BucketedActionModel;
use rf_engine::scenario::{analyze, ScenarioInput};
use std::env;
use std::fs;

const USAGE: &str = "RangeForge\n\nCommands:\n  validate <scenario.json> --model <model.json>\n  validate-model <model.json>\n  analyze <scenario.json> --model <model.json> [--format human|json] [--top N]\n\nThe bundled action model is illustrative, not empirically calibrated.";

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if let Err(error) = run(args.clone()) {
        if args.iter().any(|arg| arg == "analyze")
            && option_value(&args, "format").ok().flatten().as_deref() == Some("json")
        {
            println!(
                "{{\"error\":{}}}",
                serde_json::to_string(&error).unwrap_or_else(|_| "\"analysis failed\"".to_string())
            );
        } else {
            eprintln!("error: {error}");
        }
        std::process::exit(2);
    }
}

fn run(args: Vec<String>) -> Result<(), String> {
    if args.is_empty() || args.iter().any(|arg| arg == "--help" || arg == "-h") {
        println!("{USAGE}");
        return Ok(());
    }
    match args[0].as_str() {
        "validate-model" => {
            let model_path = positional_path(&args[1..], "model")?;
            let model = load_model(&model_path)?;
            println!(
                "valid model: {} v{} ({})",
                model.metadata().id,
                model.metadata().version,
                model.metadata().calibration
            );
            Ok(())
        }
        "validate" => {
            let scenario_path = args
                .get(1)
                .ok_or_else(|| "validate requires a scenario path".to_string())?;
            let model_path = resolve_model_path(scenario_path, &option_path(&args[2..], "model")?);
            let scenario = load_scenario(scenario_path)?;
            let _model = load_model(&model_path)?;
            scenario.validate().map_err(|error| error.to_string())?;
            println!("valid scenario: {scenario_path}");
            Ok(())
        }
        "analyze" => {
            let scenario_path = args
                .get(1)
                .ok_or_else(|| "analyze requires a scenario path".to_string())?;
            let model_path = resolve_model_path(scenario_path, &option_path(&args[2..], "model")?);
            let format = option_value(&args[2..], "format")?.unwrap_or_else(|| "human".to_string());
            let top = option_value(&args[2..], "top")?
                .map(|value| {
                    value
                        .parse::<usize>()
                        .map_err(|_| "--top must be an integer".to_string())
                })
                .transpose()?
                .unwrap_or(10);
            let scenario = load_scenario(scenario_path)?
                .validate()
                .map_err(|error| error.to_string())?;
            let model = load_model(&model_path)?;
            let report = analyze(&scenario, &model, top).map_err(|error| error.to_string())?;
            match format.as_str() {
                "human" => print_human(&report),
                "json" => println!(
                    "{}",
                    serde_json::to_string_pretty(&report).map_err(|error| error.to_string())?
                ),
                other => return Err(format!("unsupported format: {other}; use human or json")),
            }
            Ok(())
        }
        other => Err(format!("unknown command: {other}\n\n{USAGE}")),
    }
}

fn load_scenario(path: &str) -> Result<ScenarioInput, String> {
    let text = fs::read_to_string(path).map_err(|error| format!("could not read scenario {path}: {error}"))?;
    ScenarioInput::parse(&text).map_err(|error| error.to_string())
}

fn load_model(path: &str) -> Result<BucketedActionModel, String> {
    let text = fs::read_to_string(path).map_err(|error| format!("could not read model {path}: {error}"))?;
    BucketedActionModel::from_json_str(&text).map_err(|error| format!("{error:?}"))
}

fn resolve_model_path(scenario_path: &str, model_path: &str) -> String {
    let path = std::path::Path::new(model_path);
    if path.is_absolute() || path.exists() {
        model_path.to_string()
    } else {
        std::path::Path::new(scenario_path)
            .parent()
            .unwrap_or_else(|| std::path::Path::new("."))
            .join(path)
            .to_string_lossy()
            .into_owned()
    }
}

fn positional_path(args: &[String], label: &str) -> Result<String, String> {
    args.first()
        .cloned()
        .ok_or_else(|| format!("{label} command requires a path"))
}

fn option_path(args: &[String], name: &str) -> Result<String, String> {
    option_value(args, name)?.ok_or_else(|| format!("--{name} <path> is required"))
}

fn option_value(args: &[String], name: &str) -> Result<Option<String>, String> {
    let flag = format!("--{name}");
    let Some(index) = args.iter().position(|arg| arg == &flag) else {
        return Ok(None);
    };
    args.get(index + 1)
        .cloned()
        .map(Some)
        .ok_or_else(|| format!("{flag} requires a value"))
}

fn print_human(report: &rf_engine::scenario::AnalysisReport) {
    println!("RangeForge analysis");
    println!("===================");
    println!("Hero: {}    Board: {} ({})", report.hero, report.board, report.street);
    println!(
        "Model: {}{}",
        report.model_id,
        if report
            .model_metadata
            .as_ref()
            .map(|metadata| metadata.calibration.as_str())
            == Some("illustrative")
        {
            " [illustrative]"
        } else {
            ""
        }
    );
    println!();
    println!("Distribution");
    println!(
        "  prior:     support {:4}, entropy {:8.4} bits, effective hands {:8.3}",
        report.prior.support_size, report.prior.entropy_bits, report.prior.effective_hand_count
    );
    println!(
        "  posterior: support {:4}, entropy {:8.4} bits, effective hands {:8.3}",
        report.posterior.support_size, report.posterior.entropy_bits, report.posterior.effective_hand_count
    );
    println!(
        "  total variation: {:.6}    KL divergence: {:.6} bits",
        report.total_variation, report.kl_divergence_bits
    );
    println!();
    if !report.action_trace.is_empty() {
        println!("Observed actions");
        for step in &report.action_trace {
            println!(
                "  {} {:?} {:?}: likelihood {:.6}, surprisal {:.4} bits, entropy {:.4} -> {:.4}",
                step.board,
                step.decision,
                step.action,
                step.predictive_probability,
                step.surprisal_bits,
                step.entropy_before_bits,
                step.entropy_after_bits
            );
        }
        println!();
    }
    println!("Top posterior combinations");
    for combo in &report.top_combos {
        println!("  {:<5} {:.6}", combo.combo, combo.probability);
    }
    println!();
    println!("Equity");
    println!(
        "  method: {}    hero equity: {:.4}%",
        report.equity.method,
        report.equity.equity * 100.0
    );
    println!(
        "  win {:.4}%   tie {:.4}%   loss {:.4}%",
        report.equity.win_probability * 100.0,
        report.equity.tie_probability * 100.0,
        report.equity.loss_probability * 100.0
    );
    if let Some(interval) = report.equity.confidence_interval_95 {
        println!(
            "  95% interval: {:.4}% .. {:.4}%",
            interval[0] * 100.0,
            interval[1] * 100.0
        );
    }
    if let Some(info) = &report.action_information {
        println!();
        println!(
            "Action information: expected {:.4} bits",
            info.expected_information_bits
        );
        for row in &info.rows {
            println!(
                "  {:?}: probability {:.4}, information {}",
                row.action,
                row.predictive_probability,
                row.information_gain_bits
                    .map_or_else(|| "n/a".to_string(), |v| format!("{v:.4} bits"))
            );
        }
    }
    if let Some(info) = &report.next_card_information {
        println!(
            "Next-card information: expected {:.4} bits across {} cards",
            info.expected_information_bits,
            info.rows.len()
        );
    }
    for warning in &report.warnings {
        println!("warning: {warning}");
    }
}
