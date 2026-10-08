//! Synthetic training-task generation and exact-match scoring, shared by the
//! trainer and the bench suite.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

pub(crate) const TRAIN_PAIRS: usize = 2000;
pub(crate) const HELDOUT_PAIRS: usize = 100;

pub(crate) struct Example {
    pub(crate) task: &'static str,
    pub(crate) prompt: String,
    pub(crate) answer: String,
}

const AGENTS: &[&str] = &["zorp", "kit", "milu"];
const OBJECTS: &[&str] = &["blicket", "murble", "wug", "toma"];

pub(crate) fn shuffle<T>(v: &mut [T], rng: &mut StdRng) {
    for i in (1..v.len()).rev() {
        v.swap(i, rng.random_range(0..=i));
    }
}

/// Chain-of-custody state tracking: statements move objects between holders;
/// the question asks who has one object. Depth-2/3 chains for the target
/// object, one distractor statement about a second object.
fn build_state_track(rng: &mut StdRng) -> Example {
    let mut holders: Vec<&str> = AGENTS.to_vec();
    shuffle(&mut holders, rng);
    let mut objects: Vec<&str> = OBJECTS.to_vec();
    shuffle(&mut objects, rng);
    let target_obj = objects[0];
    let distractor_obj = objects[1];
    let depth = rng.random_range(2..=3usize);

    let mut lines: Vec<String> = Vec::new();
    lines.push(format!("The {target_obj} is with {}.", holders[0]));
    lines.push(format!("The {distractor_obj} is with {}.", holders[2]));
    let mut cur = holders[0];
    for step in 0..depth {
        let next = holders[1 + (step % 2)];
        lines.push(format!("{cur} gives the {target_obj} to {next}."));
        cur = next;
    }
    lines.push(format!("{} takes the {distractor_obj}.", holders[2]));
    let answer = {
        let mut c = cur.chars();
        match c.next() {
            Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
            None => String::new(),
        }
    };
    shuffle(&mut lines, rng);
    Example {
        task: "state-track",
        prompt: format!(
            "{}\nQuestion: who has the {target_obj}?\nAnswer with only the name.",
            lines.join(" ")
        ),
        answer,
    }
}

const NONSENSE: &[&str] = &[
    "dax", "wug", "zorpie", "toma", "rugg", "fep", "kiki", "gorb", "pil", "nad", "mell", "lorp",
    "seng", "tob", "cootie",
];

/// Induction with distractors: 6 relevant word->number statements plus 2-3
/// distractor statements; the 6th relevant pair is queried.
fn build_induce_d(rng: &mut StdRng) -> Example {
    let mut pool = NONSENSE.to_vec();
    shuffle(&mut pool, rng);
    let chosen: Vec<&str> = pool[..6].to_vec();
    let distractors: Vec<&str> = pool[6..9].to_vec();
    let numbers: Vec<i64> = (0..6).map(|_| rng.random_range(1..100)).collect();
    let mut stmts: Vec<String> = chosen
        .iter()
        .zip(&numbers)
        .map(|(w, n)| format!("{w} is {n}"))
        .collect();
    let n_distract = rng.random_range(2..=3usize);
    for d in distractors.iter().take(n_distract) {
        stmts.push(format!("{d} is red."));
    }
    shuffle(&mut stmts, rng);
    let query = chosen[5];
    let answer = numbers[5].to_string();
    Example {
        task: "induce-d",
        prompt: format!(
            "{}.\nQuestion: {query} is?\nAnswer with only the number.",
            stmts.join(". ")
        ),
        answer,
    }
}

pub(crate) fn build_dataset() -> (Vec<Example>, Vec<Example>) {
    let mut rng = StdRng::seed_from_u64(777);
    let mut train = Vec::new();
    for _ in 0..TRAIN_PAIRS {
        train.push(build_state_track(&mut rng));
        train.push(build_induce_d(&mut rng));
    }
    // Held-out hygiene: the generators are small-combinatorial-space tasks
    // (state-track's distinct-prompt space is ~1e5, with 4k train draws), so
    // a plain sequential split leaves ~3% of held-out state-track prompts as
    // literal training prompts. Redraw any held-out example whose prompt
    // string duplicates a training example (or an earlier held-out one).
    let mut seen: std::collections::HashSet<String> =
        train.iter().map(|e| e.prompt.clone()).collect();
    let mut heldout = Vec::new();
    let mut n_replaced = 0usize;
    for _ in 0..HELDOUT_PAIRS {
        let mut st = build_state_track(&mut rng);
        while seen.contains(&st.prompt) {
            n_replaced += 1;
            st = build_state_track(&mut rng);
        }
        seen.insert(st.prompt.clone());
        heldout.push(st);
        let mut id = build_induce_d(&mut rng);
        while seen.contains(&id.prompt) {
            n_replaced += 1;
            id = build_induce_d(&mut rng);
        }
        seen.insert(id.prompt.clone());
        heldout.push(id);
    }
    if n_replaced > 0 {
        eprintln!(
            "dataset: redrew {n_replaced} held-out prompts that duplicated training examples"
        );
    }
    (train, heldout)
}

pub(crate) fn first_nonempty_line(text: &str) -> &str {
    text.lines()
        .find(|l| !l.trim().is_empty())
        .map(str::trim)
        .unwrap_or("")
}

pub(crate) fn last_int(s: &str) -> Option<i64> {
    let mut last = None;
    let mut cur: Option<i64> = None;
    for c in s.chars() {
        if let Some(d) = c.to_digit(10) {
            cur = Some(cur.unwrap_or(0).saturating_mul(10).saturating_add(d as i64));
        } else if let Some(v) = cur.take() {
            last = Some(v);
        }
    }
    if let Some(v) = cur {
        last = Some(v);
    }
    last
}

/// Exact-match scoring for the two task families.
pub(crate) fn score(task: &str, answer: &str, text: &str) -> bool {
    let line = first_nonempty_line(text).to_lowercase();
    match task {
        "state-track" => {
            let words: Vec<String> = line
                .chars()
                .map(|c| if c.is_alphanumeric() { c } else { ' ' })
                .collect::<String>()
                .split_whitespace()
                .map(str::to_string)
                .collect();
            let answer_lc = answer.to_lowercase();
            words.contains(&answer_lc)
        }
        _ => last_int(&line) == answer.parse::<i64>().ok(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn last_int_extracts_trailing_number() {
        assert_eq!(last_int("42"), Some(42));
        assert_eq!(last_int("abc 42 def 7"), Some(7));
        assert_eq!(last_int("value: 0012."), Some(12));
        assert_eq!(last_int("no digits here"), None);
        assert_eq!(last_int(""), None);
    }

    #[test]
    fn first_nonempty_line_trims() {
        assert_eq!(first_nonempty_line("  \n  hello \n  world"), "hello");
        assert_eq!(first_nonempty_line(""), "");
    }

    #[test]
    fn score_matches_task_families() {
        assert!(score("induce-d", "42", "42\n"));
        assert!(!score("induce-d", "42", "the answer is 41"));
        assert!(score("state-track", "Zorp", "Zorp."));
        assert!(!score("state-track", "Zorp", "Kit."));
    }

    #[test]
    fn shuffle_is_a_permutation() {
        let mut rng = StdRng::seed_from_u64(1);
        let mut v: Vec<usize> = (0..64).collect();
        shuffle(&mut v, &mut rng);
        let mut sorted = v.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..64).collect::<Vec<usize>>());
    }
}
