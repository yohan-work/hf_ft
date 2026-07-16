const names = { payment: "결제", cancellation_refund: "취소·환불", delivery: "배송", exchange_return: "교환·반품", account: "계정", technical_issue: "기술 오류", product_general: "상품·일반" };
const form = document.querySelector("#predict-form");
const inquiry = document.querySelector("#inquiry");
const count = document.querySelector("#char-count");
const error = document.querySelector("#predict-error");
const button = document.querySelector("#predict-button");

function updateCount() { count.textContent = `${inquiry.value.length} / 1000`; }
function renderPrediction(data) {
  document.querySelector("#result-label-name").textContent = data.label_name;
  document.querySelector("#result-label-code").textContent = data.label;
  document.querySelector("#result-confidence").textContent = `${(data.confidence * 100).toFixed(1)}%`;
  document.querySelector("#result-state").textContent = "COMPLETE";
  const list = document.querySelector("#score-list"); list.replaceChildren();
  data.scores.forEach((score) => {
    const row = document.createElement("div"); row.className = "score-row";
    row.innerHTML = `<span>${names[score.label] || score.label}</span><div class="score-bar"><i style="width:${score.score * 100}%"></i></div><b>${(score.score * 100).toFixed(1)}</b>`;
    list.append(row);
  });
}
async function runPrediction(event) {
  event.preventDefault(); error.textContent = ""; button.disabled = true; button.querySelector("span").textContent = "분석 중";
  try {
    const response = await fetch("/predict", { method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({text: inquiry.value}) });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error?.details?.[0]?.msg || "예측 요청에 실패했습니다.");
    renderPrediction(data);
  } catch (issue) { error.textContent = issue.message; }
  finally { button.disabled = false; button.querySelector("span").textContent = "예측 실행"; }
}
function metric(label, value, note, direction="") { return `<div class="metric-card"><span class="label">${label}</span><strong class="${direction}">${value}</strong><small>${note}</small></div>`; }
function renderMatrix(metrics, labels) {
  const maximum = Math.max(...metrics.confusion_matrix.flat(), 1); let html = "<table class='matrix'><thead><tr><th>ACT↓ / PRED→</th>";
  labels.forEach((label) => html += `<th>${label.slice(0, 5)}</th>`); html += "</tr></thead><tbody>";
  metrics.confusion_matrix.forEach((row, index) => { html += `<tr><th>${labels[index].slice(0, 5)}</th>`; row.forEach((value) => html += `<td style='--shade:${value / maximum * .72}'>${value}</td>`); html += "</tr>"; });
  document.querySelector("#matrix-wrap").innerHTML = html + "</tbody></table>";
}
function renderLab(data) {
  const transformer = data.fine_tuned_klue_roberta; const baseline = data.tfidf_logistic_regression;
  document.querySelector("#metric-cards").innerHTML = [
    metric("TRANSFORMER MACRO F1", transformer.metrics.macro_f1.toFixed(3), "fixed Test · 21 inquiries", "up"),
    metric("BASELINE MACRO F1", baseline.metrics.macro_f1.toFixed(3), "TF-IDF + Logistic Regression"),
    metric("DELTA", `+${(transformer.metrics.macro_f1 - baseline.metrics.macro_f1).toFixed(3)}`, "Transformer minus baseline", "up"),
    metric("INFERENCE", `${transformer.inference_ms_per_text.toFixed(1)} ms`, `${(transformer.model_size_bytes / 1e6).toFixed(1)} MB · CPU`)
  ].join("");
  renderMatrix(transformer.metrics, data.label_order);
  const classList = document.querySelector("#class-metrics"); classList.replaceChildren();
  Object.entries(transformer.metrics.per_class).forEach(([label, values]) => {
    const row = document.createElement("div"); row.className = `class-row ${values.f1 < .5 ? "low" : ""}`;
    row.innerHTML = `<span>${names[label]}</span><div class="f1bar"><i style="width:${values.f1 * 100}%"></i></div><strong>${values.f1.toFixed(2)}</strong>`; classList.append(row);
  });
  const errors = transformer.all_errors; document.querySelector("#error-count").textContent = `${errors.length} errors / ${data.test_size} test cases`;
  const errorList = document.querySelector("#error-list"); errorList.replaceChildren();
  errors.forEach((item) => { const card = document.createElement("article"); card.className = "error-card"; card.innerHTML = `<p></p><div class="error-route"><span>actual </span><b>${names[item.true_label]}</b><span> → predicted </span><b>${names[item.predicted_label]}</b><div class="error-confidence">confidence ${(item.confidence * 100).toFixed(1)}%</div></div>`; card.querySelector("p").textContent = item.text; errorList.append(card); });
}
async function initialize() {
  updateCount(); inquiry.addEventListener("input", updateCount); form.addEventListener("submit", runPrediction);
  try { const health = await fetch("/health").then((r) => r.json()); document.querySelector("#health-status").textContent = "model loaded"; document.querySelector("#model-version").textContent = health.model_version; document.querySelector("#device-name").textContent = health.device; }
  catch { document.querySelector("#health-status").textContent = "connection failed"; }
  try { renderLab(await fetch("/lab/metrics").then((r) => r.json())); } catch { document.querySelector("#metric-cards").innerHTML = metric("EVALUATION", "—", "artifacts unavailable"); }
}
initialize();
