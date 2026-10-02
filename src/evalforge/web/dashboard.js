'use strict';
const history = document.querySelector('#history');
const detail = document.querySelector('#detail');
const notice = document.querySelector('#notice');
let selected = null;
let requestSequence = 0;
function node(tag, text, className) {
  const element = document.createElement(tag);
  if (text !== undefined) element.textContent = text;
  if (className) element.className = className;
  return element;
}
function metricLabel(value) { return value.replace(/_/g, ' ').replace(/^./, letter => letter.toUpperCase()); }
function badge(status) { return node('span', status, `badge ${status}`); }
function number(value) { return value == null ? '—' : Number(value).toFixed(3); }
function metricValue(value, metric, delta = false) {
  if (value == null) return '—';
  if (['latency_ms', 'p95_latency_ms'].includes(metric)) return `${value.toFixed(1)} ms`;
  if (metric === 'cost') return `$${value.toFixed(4)}`;
  return `${(value * 100).toFixed(1)}${delta ? ' pp' : '%'}`;
}
async function get(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error('Could not read this experiment. Check that its saved files are complete.');
  return response.json();
}
function answerPanel(label, row, run) {
  const box = node('div', undefined, 'answer');
  box.append(node('h3', label), node('p', row.target.output ?? row.target.error ?? 'No answer saved', 'answer-text'));
  if (run.target_config.provider?.model) box.append(node('p', `Configured model: ${run.target_config.provider.model}`, 'muted'));
  box.append(node('p', `${run.target_config.kind ?? 'Unknown'} target · ${row.target.latency_ms.toFixed(1)} ms`, 'muted'));
  if (row.target.error) box.append(node('p', `Execution error: ${row.target.error}`));
  const evaluations = node('div', undefined, 'evaluations');
  for (const evaluation of row.evaluations) {
    const line = node('div', undefined, 'evaluation');
    line.append(node('strong', `${metricLabel(evaluation.metric)} · ${metricValue(evaluation.score, evaluation.metric)}`), badge(evaluation.status), node('span', evaluation.explanation, 'explanation'));
    evaluations.append(line);
    if (Array.isArray(evaluation.metadata.evidence) && evaluation.metadata.evidence.length) line.append(node('p', `Judge evidence: ${evaluation.metadata.evidence.join('; ')}`, 'muted judge-evidence'));
  }
  box.append(evaluations);
  const trace = row.target.retrieval;
  if (trace) {
    const evidence = node('details');
    evidence.append(node('summary', `Retrieved evidence · ${trace.passages.length} passages`));
    evidence.append(node('p', `Corpus: ${trace.corpus_hash}`, 'source'));
    if (!trace.passages.length) evidence.append(node('p', 'No matching passages were retrieved.'));
    for (const passage of trace.passages) {
      const block = node('div', undefined, 'passage');
      block.append(node('strong', `${passage.rank}. ${passage.document_id} · ${passage.heading}`), node('p', passage.text), node('p', `${passage.chunk_id} · Search score ${number(passage.score)}`, 'source'));
      evidence.append(block);
    }
    evidence.append(node('p', `Citations: ${trace.citations.join(', ') || 'None'}`, 'source'));
    box.append(evidence);
  }
  return box;
}
function showExperiment(experiment) {
  detail.replaceChildren();
  const {baseline, candidate, comparison} = experiment;
  const overview = node('div', undefined, 'card overview');
  const title = node('div', undefined, 'title-row');
  title.append(node('h2', candidate.dataset_name), badge(comparison.status));
  overview.append(title, node('p', `${baseline.target_name} → ${candidate.target_name}`, 'run-description'), node('p', `${candidate.cases.length} questions · ${new Date(candidate.created_at).toLocaleString()}`, 'muted'));
  const kind = candidate.target_config.kind;

  const gates = node('div', undefined, 'gates');
  for (const gate of comparison.gates) {
    const card = node('div', undefined, 'gate');
    const heading = node('div', undefined, 'gate-title');
    heading.append(node('strong', `${metricLabel(gate.rule.metric)}${gate.rule.category ? ` · ${gate.rule.category}` : ''}`), badge(gate.status));
    const values = node('div', undefined, 'gate-values');
    const before = node('span', metricValue(gate.baseline, gate.rule.metric), 'before');
    before.setAttribute('aria-label', `Baseline: ${metricValue(gate.baseline, gate.rule.metric)}`);
    const after = node('strong', metricValue(gate.candidate, gate.rule.metric));
    after.setAttribute('aria-label', `Candidate: ${metricValue(gate.candidate, gate.rule.metric)}`);
    values.append(before, node('span', '→', 'arrow'), after);
    const explanation = node('details');
    explanation.append(node('summary', 'Decision details'), node('p', `${gate.rule.severity === 'warn' ? 'Warning check' : 'Required check'} · ${gate.explanation}`));
    card.append(heading, values, node('p', `${metricValue(gate.delta, gate.rule.metric, true)} change from baseline`, 'delta'), explanation);
    gates.append(card);
  }
  overview.append(gates);
  overview.append(node('p', candidate.target_config.evidence_mode === 'simulated' ? 'Simulated provider responses for UI verification. No real AI calls were made.' : kind === 'rag' ? 'RAG target: answers use the configured model. Tests may also produce saved runs with simulated provider responses; inspect how the run was created.' : kind === 'mock' ? 'Fixture responses: these answers were predefined, not generated by AI.' : kind === 'local' ? 'Local application · These answers may come from rules or a model. Check the target configuration.' : 'Model target: answers use the configured provider.', 'provenance'));
  detail.append(overview);
  const filters = node('div', undefined, 'filters');
  const filter = node('select'); filter.setAttribute('aria-label', 'Filter questions');
  for (const [value, label] of [['failed', 'Failures and errors'], ['all', 'All questions']]) {
    const option = node('option', label); option.value = value; filter.append(option);
  }
  filters.append(node('h2', 'Question details'), filter); detail.append(filters);
  const cases = node('div'); detail.append(cases);
  const previous = new Map(baseline.cases.map(row => [row.case.id, row]));
  function renderCases() {
    cases.replaceChildren();
    for (const row of candidate.cases) {
      const failed = row.target.status === 'ERROR' || row.evaluations.some(e => ['FAIL', 'ERROR', 'UNKNOWN'].includes(e.status));
      const before = previous.get(row.case.id);
      const baselineFailed = before && (before.target.status === 'ERROR' || before.evaluations.some(e => ['ERROR', 'UNKNOWN'].includes(e.status)));
      if (filter.value === 'failed' && !failed && !baselineFailed) continue;
      const card = node('article', undefined, 'card');
      const head = node('div', undefined, 'case-head');
      const meta = node('div', undefined, 'case-meta');
      meta.append(node('span', row.case.category), node('span', '·'), node('span', row.case.id));
      if (row.case.critical) meta.append(node('span', 'Critical', 'critical'));
      head.append(meta, node('p', typeof row.case.input === 'string' ? row.case.input : JSON.stringify(row.case.input), 'case-question'));
      if (row.case.reference_answer != null) {
        const reference = node('details', undefined, 'reference');
        reference.append(node('summary', 'Expected answer'), node('p', `${row.case.reference_answer} (evaluation only)`));
        head.append(reference);
      }
      card.append(head);
      const panels = node('div', undefined, 'answers');
      if (before) panels.append(answerPanel('BASELINE', before, baseline));
      panels.append(answerPanel('CANDIDATE', row, candidate)); card.append(panels); cases.append(card);
    }
    if (!cases.childElementCount) cases.append(node('p', 'No question-level failures in this view. Aggregate gates may still fail; inspect their explanations above.'));
  }
  filter.addEventListener('change', renderCases); renderCases();
}
async function selectExperiment(id) {
  selected = id;
  const sequence = ++requestSequence;
  for (const button of history.querySelectorAll('button')) {
    const active = button.dataset.id === id;
    button.classList.toggle('selected', active);
    button.setAttribute('aria-pressed', String(active));
  }
  try {
    const experiment = await get(`/api/experiments/${encodeURIComponent(id)}`);
    if (sequence !== requestSequence) return;
    showExperiment(experiment);
  } catch (error) { if (sequence === requestSequence) notice.textContent = error.message; }
}
async function refresh() {
  try {
    const response = await get('/api/experiments');
    notice.textContent = response.invalid_artifacts ? `${response.invalid_artifacts} incomplete or invalid artifact folders were skipped.` : '';
    history.replaceChildren();
    for (const run of response.experiments) {
      const button = node('button', undefined, 'run'); button.dataset.id = run.directory;
      const top = node('div', undefined, 'run-top');
      top.append(node('strong', run.dataset), badge(run.status));
      button.append(top, node('span', `${run.baseline} → ${run.candidate}`, 'muted'), node('span', new Date(run.created_at).toLocaleString(), 'muted'));
      button.addEventListener('click', () => selectExperiment(run.directory)); history.append(button);
    }
    if (response.experiments.length) {
      const id = response.experiments.some(run => run.directory === selected) ? selected : response.experiments[0].directory;
      await selectExperiment(id);
    } else { history.append(node('p', 'No saved experiments yet.', 'muted')); }
  } catch (error) { notice.textContent = error.message; }
}
document.querySelector('#refresh').addEventListener('click', refresh);
refresh();
