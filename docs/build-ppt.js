// ============================================================
// Behavioral Fraud Analytics — Master Thesis Presentation
// Koutsompinas Konstantinos | NKUA MSc | February 2026
// VERSION 2: 20 slides — added Literature, Data Split, SHAP
// ============================================================
const pptxgen = require("pptxgenjs");
const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.title   = "Behavioral Fraud Analytics";
pres.author  = "Koutsompinas Konstantinos";

// ─────────────────────────────────────────────────────────────
// DESIGN SYSTEM
// ─────────────────────────────────────────────────────────────
const C = {
  navyDark  : "0A1628",
  navy      : "0D2137",
  navyMid   : "1A3A5C",
  teal      : "0891B2",
  tealBright: "06B6D4",
  tealPale  : "DBEAFE",
  bgPage    : "EEF4FB",
  bgCard    : "FFFFFF",
  bgCardAlt : "F1F7FF",
  textDark  : "1E293B",
  textMid   : "475569",
  textMuted : "94A3B8",
  amber     : "D97706",
  amberPale : "FEF3C7",
  green     : "059669",
  greenPale : "D1FAE5",
  red       : "DC2626",
  redPale   : "FEE2E2",
  purple    : "7C3AED",
  purplePale: "EDE9FE",
  white     : "FFFFFF",
};

// Slide dimensions (16:9)
const W = 10, H = 5.625;
const AW  = 0.07;   // left accent bar width
const HH  = 0.82;   // header height
const M   = 0.42;   // margin

// ─────────────────────────────────────────────────────────────
// HELPERS
// ─────────────────────────────────────────────────────────────
// Fresh shadow — prevents pptxgenjs object-mutation bug
const sh  = () => ({ type:"outer", blur:8,  offset:2, angle:135, color:"000000", opacity:0.10 });
const shH = () => ({ type:"outer", blur:16, offset:4, angle:135, color:"000000", opacity:0.14 });

/** Add dark-navy full-slide background + corner decorations */
function darkBg(sl) {
  sl.background = { color: C.navyDark };
  // Top-right corner lines
  sl.addShape(pres.shapes.RECTANGLE, { x:W-2.0, y:0,      w:2.0, h:0.04, fill:{color:C.teal,       transparency:50}, line:{width:0} });
  sl.addShape(pres.shapes.RECTANGLE, { x:W-0.04, y:0,     w:0.04,h:2.0,  fill:{color:C.teal,       transparency:50}, line:{width:0} });
  // Bottom-left corner lines
  sl.addShape(pres.shapes.RECTANGLE, { x:0, y:H-0.04,     w:2.0, h:0.04, fill:{color:C.tealBright, transparency:60}, line:{width:0} });
  sl.addShape(pres.shapes.RECTANGLE, { x:0, y:H-2.0,      w:0.04,h:2.0,  fill:{color:C.tealBright, transparency:60}, line:{width:0} });
}

/** Add light-background slide with accent bar + header band + title */
function lightSlide(sl, title, num) {
  sl.background = { color: C.bgPage };
  sl.addShape(pres.shapes.RECTANGLE, { x:0,  y:0, w:AW,    h:H,  fill:{color:C.teal},     line:{width:0} });
  sl.addShape(pres.shapes.RECTANGLE, { x:AW, y:0, w:W-AW,  h:HH, fill:{color:C.navyDark}, line:{width:0} });
  sl.addText(title, {
    x:AW+0.28, y:0, w:W-AW-1.1, h:HH,
    fontSize:22, bold:true, color:C.white, valign:"middle",
    fontFace:"Calibri", margin:0
  });
  if (num) {
    sl.addText(String(num).padStart(2,"0"), {
      x:W-0.68, y:0, w:0.58, h:HH,
      fontSize:16, bold:true, color:C.tealBright,
      align:"right", valign:"middle", fontFace:"Calibri", margin:0
    });
  }
}

/** White card with left color accent bar */
function accentCard(sl, x, y, w, h, accent, shadow=true) {
  sl.addShape(pres.shapes.RECTANGLE, { x, y, w, h,    fill:{color:C.bgCard},   line:{color:"E2E8F0",width:1}, shadow:shadow?sh():undefined });
  sl.addShape(pres.shapes.RECTANGLE, { x, y, w:0.055,h, fill:{color:accent},   line:{width:0} });
}

/** Plain white card */
function card(sl, x, y, w, h, fill, shadow=true) {
  sl.addShape(pres.shapes.RECTANGLE, { x, y, w, h,
    fill:{color:fill||C.bgCard}, line:{color:"E2E8F0",width:1},
    shadow:shadow?sh():undefined
  });
}

/** Colored metric callout box */
function metricBox(sl, x, y, w, h, val, lbl, bg, fg) {
  sl.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill:{color:bg}, line:{width:0}, shadow:sh() });
  sl.addText(val, { x, y:y+0.05,     w, h:h*0.58, fontSize:h<0.9?20:32, bold:true, color:fg||C.white, align:"center", valign:"middle", fontFace:"Calibri", margin:0 });
  sl.addText(lbl, { x, y:y+h*0.60,   w, h:h*0.38, fontSize:9.5, color:fg||C.white, align:"center", valign:"top",    fontFace:"Calibri", margin:0 });
}

/** Step circle with number */
function stepCircle(sl, cx, cy, r, num, bg) {
  sl.addShape(pres.shapes.OVAL, { x:cx-r, y:cy-r, w:r*2, h:r*2, fill:{color:bg}, line:{width:0} });
  sl.addText(String(num), { x:cx-r, y:cy-r, w:r*2, h:r*2, fontSize:18, bold:true, color:C.white, align:"center", valign:"middle", fontFace:"Calibri", margin:0 });
}

// ─────────────────────────────────────────────────────────────
// SLIDE 01 — TITLE
// ─────────────────────────────────────────────────────────────
{
  const sl = pres.addSlide();
  darkBg(sl);

  // Vertical left accent stripe
  sl.addShape(pres.shapes.RECTANGLE, { x:0, y:0, w:0.12, h:H, fill:{color:C.teal}, line:{width:0} });

  // Background subtle horizontal lines (texture)
  for (let i=0; i<8; i++) {
    sl.addShape(pres.shapes.RECTANGLE, {
      x:0.12, y:0.7*i+0.1, w:W-0.12, h:0.01,
      fill:{color:C.teal, transparency:88}, line:{width:0}
    });
  }

  // Label chip
  sl.addShape(pres.shapes.RECTANGLE, { x:0.55, y:0.55, w:1.85, h:0.30, fill:{color:C.teal}, line:{width:0} });
  sl.addText("MSc THESIS PRESENTATION", {
    x:0.55, y:0.55, w:1.85, h:0.30,
    fontSize:7.5, bold:true, color:C.white, align:"center", valign:"middle", fontFace:"Calibri", margin:0
  });

  // Main title
  sl.addText("BEHAVIORAL\nFRAUD ANALYTICS", {
    x:0.55, y:1.05, w:6.8, h:2.35,
    fontSize:46, bold:true, color:C.white, align:"left", valign:"middle",
    fontFace:"Calibri", lineSpacingMultiple:1.05, margin:0
  });

  // Subtitle
  sl.addText("Machine Learning for Fraud Detection in Online Financial Transactions", {
    x:0.55, y:3.45, w:7.2, h:0.55,
    fontSize:14, color:C.tealBright, align:"left", valign:"middle",
    fontFace:"Calibri", margin:0
  });

  // Divider line
  sl.addShape(pres.shapes.LINE, { x:0.55, y:4.1, w:6.8, h:0, line:{color:C.teal, width:1.5} });

  // Author info
  sl.addText([
    { text:"Koutsompinas Konstantinos", options:{bold:true, breakLine:true} },
    { text:"Supervisor: Athanasios Argyriou", options:{breakLine:true} },
    { text:"National & Kapodistrian University of Athens  |  Department of Economics  |  February 2026" },
  ], {
    x:0.55, y:4.2, w:7.5, h:1.0,
    fontSize:11, color:"94A3B8", fontFace:"Calibri", align:"left", valign:"top", margin:0
  });

  // Right side: dramatic stat callout
  sl.addShape(pres.shapes.RECTANGLE, { x:7.9, y:1.3, w:1.75, h:1.8, fill:{color:C.navyMid}, line:{color:C.teal, width:1.5}, shadow:shH() });
  sl.addText("3.5%", { x:7.9, y:1.5, w:1.75, h:0.8, fontSize:36, bold:true, color:C.tealBright, align:"center", valign:"middle", fontFace:"Calibri", margin:0 });
  sl.addText("fraud rate\nin IEEE-CIS dataset", { x:7.9, y:2.3, w:1.75, h:0.65, fontSize:9.5, color:"94A3B8", align:"center", valign:"top", fontFace:"Calibri", margin:0 });

  sl.addShape(pres.shapes.RECTANGLE, { x:7.9, y:3.3, w:1.75, h:1.5, fill:{color:C.navyMid}, line:{color:C.navyMid, width:1}, shadow:sh() });
  sl.addText("590K+", { x:7.9, y:3.45, w:1.75, h:0.7, fontSize:30, bold:true, color:C.white, align:"center", valign:"middle", fontFace:"Calibri", margin:0 });
  sl.addText("transactions\nanalyzed", { x:7.9, y:4.15, w:1.75, h:0.55, fontSize:9.5, color:"94A3B8", align:"center", valign:"top", fontFace:"Calibri", margin:0 });
}

// ─────────────────────────────────────────────────────────────
// SLIDE 02 — PRESENTATION ROADMAP (7 sections)
// ─────────────────────────────────────────────────────────────
{
  const sl = pres.addSlide();
  darkBg(sl);

  sl.addText("PRESENTATION ROADMAP", {
    x:0.5, y:0.25, w:9.0, h:0.45,
    fontSize:10, bold:true, color:C.teal, align:"center", valign:"middle",
    charSpacing:5, fontFace:"Calibri", margin:0
  });
  sl.addText("What We'll Cover Today", {
    x:0.5, y:0.66, w:9.0, h:0.58,
    fontSize:28, bold:true, color:C.white, align:"center", valign:"middle",
    fontFace:"Calibri", margin:0
  });

  const sections = [
    ["01", "Challenge & Context",       "The fraud detection problem and why it's hard"],
    ["02", "Literature & Model Choice", "ML landscape review and gradient boosting rationale"],
    ["03", "Research Design",           "Four research questions guiding the study"],
    ["04", "Dataset & Methodology",     "IEEE-CIS data, pipeline, feature engineering, data split"],
    ["05", "Experimental Results",      "Four configurations across three models"],
    ["06", "Synthesis & Insights",      "Cross-experiment patterns and what they mean"],
    ["07", "Conclusions",               "Key takeaways and future research directions"],
  ];

  // 4 items left column, 3 items right column
  const cols = [[0,1,2,3],[4,5,6]];
  const colX = [0.45, 5.3];
  const startY = 1.42, cardW = 4.35, cardH = 0.76, gapY = 0.82;

  cols.forEach((indices, ci) => {
    indices.forEach((idx, ri) => {
      const [num, title, desc] = sections[idx];
      const cx = colX[ci], cy = startY + ri * gapY;

      sl.addShape(pres.shapes.RECTANGLE, { x:cx, y:cy, w:cardW, h:cardH, fill:{color:C.navy}, line:{color:C.navyMid, width:1}, shadow:sh() });
      // Number
      sl.addShape(pres.shapes.RECTANGLE, { x:cx, y:cy, w:0.50, h:cardH, fill:{color:C.teal}, line:{width:0} });
      sl.addText(num, { x:cx, y:cy, w:0.50, h:cardH, fontSize:15, bold:true, color:C.white, align:"center", valign:"middle", fontFace:"Calibri", margin:0 });
      // Title + desc
      sl.addText(title, { x:cx+0.60, y:cy+0.05, w:cardW-0.70, h:0.30, fontSize:12.5, bold:true, color:C.white, valign:"middle", fontFace:"Calibri", margin:0 });
      sl.addText(desc,  { x:cx+0.60, y:cy+0.40, w:cardW-0.70, h:0.30, fontSize:9.5, color:"94A3B8", valign:"top",    fontFace:"Calibri", margin:0 });
    });
  });
}

// ─────────────────────────────────────────────────────────────
// SLIDE 03 — THE FRAUD DETECTION CHALLENGE
// ─────────────────────────────────────────────────────────────
{
  const sl = pres.addSlide();
  lightSlide(sl, "The Fraud Detection Challenge", 3);

  const challenges = [
    { color:C.red,    title:"Extreme Class Imbalance",        desc:"Fraudulent transactions represent only ~3.5% of all activity. Standard models learn to predict the majority class and miss fraud entirely." },
    { color:C.amber,  title:"Continuously Evolving Tactics",  desc:"Fraudsters constantly adapt their methods to evade detection, making rule-based systems obsolete within months." },
    { color:C.teal,   title:"Real-Time Constraints",          desc:"Detection decisions must be made in milliseconds at scale — complex models must also be computationally practical." },
    { color:C.purple, title:"Interpretability vs. Accuracy",  desc:"Complex ensemble models achieve the best results, but financial regulators and risk teams demand explainable decisions." },
  ];

  const cy0 = 1.0, ch = 0.97, cg = 0.07;
  challenges.forEach(({ color, title, desc }, i) => {
    const y = cy0 + i*(ch+cg);
    accentCard(sl, AW+M, y, 5.8, ch, color);
    sl.addText(title, { x:AW+M+0.22, y:y+0.06, w:5.4, h:0.35, fontSize:13, bold:true, color:C.textDark, valign:"middle", fontFace:"Calibri", margin:0 });
    sl.addText(desc,  { x:AW+M+0.22, y:y+0.42, w:5.4, h:0.50, fontSize:10.5, color:C.textMid, valign:"top",    fontFace:"Calibri", margin:0 });
  });

  // Right panel: stat callouts
  const rx = 6.85, rw = 2.88;

  // Big stat: fraud rate
  sl.addShape(pres.shapes.RECTANGLE, { x:rx, y:1.0, w:rw, h:1.6, fill:{color:C.navyDark}, line:{color:C.teal, width:1.5}, shadow:shH() });
  sl.addText("3.5%", { x:rx, y:1.1, w:rw, h:0.85, fontSize:50, bold:true, color:C.tealBright, align:"center", valign:"middle", fontFace:"Calibri", margin:0 });
  sl.addText("fraud rate in IEEE-CIS", { x:rx, y:1.92, w:rw, h:0.55, fontSize:10.5, color:"94A3B8", align:"center", valign:"top", fontFace:"Calibri", margin:0 });

  metricBox(sl, rx,         2.78, rw*0.48, 1.05, "590K+", "Transactions",    C.teal,    C.white);
  metricBox(sl, rx+rw*0.52, 2.78, rw*0.48, 1.05, "434",   "Features",        C.navyMid, C.white);
  metricBox(sl, rx,         4.0,  rw*0.48, 1.05, "3",     "ML Models",       C.purple,  C.white);
  metricBox(sl, rx+rw*0.52, 4.0,  rw*0.48, 1.05, "4",     "Research\nQuestions", C.amber, C.white);
}

// ─────────────────────────────────────────────────────────────
// SLIDE 04 — RESEARCH QUESTIONS
// ─────────────────────────────────────────────────────────────
{
  const sl = pres.addSlide();
  lightSlide(sl, "Research Questions", 4);

  const rqs = [
    { n:"RQ 1", color:C.teal,
      q:"Model Discrimination",
      t:"How effectively do gradient boosting models (XGBoost, LightGBM, CatBoost) discriminate fraudulent from legitimate transactions under ROC-AUC evaluation?" },
    { n:"RQ 2", color:C.purple,
      q:"Impact of Downsampling",
      t:"How does majority-class downsampling (1:5 ratio) affect model performance compared to training on the original imbalanced data distribution?" },
    { n:"RQ 3", color:C.green,
      q:"Feature Set Reduction",
      t:"Can a reduced feature set, selected through cross-model feature importance agreement, preserve the discriminatory power of the full feature space?" },
    { n:"RQ 4", color:C.amber,
      q:"Threshold Tuning Effects",
      t:"How does altering the decision threshold change fraud detection behavior — specifically the recall–precision trade-off relevant to operational deployment?" },
  ];

  const x0 = AW+M, y0 = 1.05, cw = 4.38, ch = 1.92, gx = 0.22, gy = 0.22;
  rqs.forEach(({ n, color, q, t }, i) => {
    const cx = x0 + (i%2)*(cw+gx);
    const cy = y0 + Math.floor(i/2)*(ch+gy);

    card(sl, cx, cy, cw, ch);
    // Top color bar inside card
    sl.addShape(pres.shapes.RECTANGLE, { x:cx, y:cy, w:cw, h:0.42, fill:{color}, line:{width:0} });
    sl.addText(n, { x:cx+0.15, y:cy, w:1.0, h:0.42, fontSize:12, bold:true, color:C.white, valign:"middle", fontFace:"Calibri", margin:0 });
    sl.addText(q, { x:cx+1.1,  y:cy, w:cw-1.2, h:0.42, fontSize:12, bold:true, color:C.white, valign:"middle", fontFace:"Calibri", margin:0 });
    sl.addText(t, { x:cx+0.20, y:cy+0.50, w:cw-0.35, h:1.32, fontSize:11.5, color:C.textMid, valign:"top", fontFace:"Calibri", margin:0 });
  });
}

// ─────────────────────────────────────────────────────────────
// SLIDE 05 — ML LITERATURE LANDSCAPE & WHY GRADIENT BOOSTING
// ─────────────────────────────────────────────────────────────
{
  const sl = pres.addSlide();
  lightSlide(sl, "Literature Landscape: ML Approaches to Fraud Detection", 5);

  const approaches = [
    { color:C.red,    tag:"Traditional Supervised",
      models:"Logistic Regression, SVM, Decision Trees",
      pro:"Interpretable, fast, well-understood",
      con:"Limited capacity to model complex interactions; poor recall on severe imbalance" },
    { color:C.amber,  tag:"Anomaly Detection",
      models:"Isolation Forest, One-Class SVM, LOF",
      pro:"Works without labeled fraud — useful when labels are scarce",
      con:"High false positive rates; not optimized for known fraud patterns in labeled data" },
    { color:C.purple, tag:"Deep Learning",
      models:"LSTM, Autoencoders, Transformers",
      pro:"Powerful on sequential/unstructured data; learns latent representations",
      con:"Requires large datasets, high compute, difficult to interpret for compliance" },
    { color:C.teal,   tag:"Gradient Boosting Ensembles",
      models:"XGBoost, LightGBM, CatBoost",
      pro:"State-of-the-art on tabular data; handles missing values, imbalance, and scale",
      con:"Ensemble complexity requires SHAP or similar for interpretability" },
  ];

  const x0 = AW+M, y0 = 1.00, cw = 4.38, ch = 1.65, gx = 0.22, gy = 0.18;
  approaches.forEach(({ color, tag, models, pro, con }, i) => {
    const cx = x0 + (i%2)*(cw+gx);
    const cy = y0 + Math.floor(i/2)*(ch+gy);

    card(sl, cx, cy, cw, ch);
    // Top color strip
    sl.addShape(pres.shapes.RECTANGLE, { x:cx, y:cy, w:cw, h:0.38, fill:{color}, line:{width:0} });
    sl.addText(tag, { x:cx+0.14, y:cy, w:cw-0.20, h:0.38, fontSize:12, bold:true, color:C.white, valign:"middle", fontFace:"Calibri", margin:0 });
    // Models in italic
    sl.addText(models, { x:cx+0.14, y:cy+0.42, w:cw-0.22, h:0.28, fontSize:10, color:C.textMuted, valign:"top", fontFace:"Calibri", margin:0 });
    // Pro
    sl.addShape(pres.shapes.RECTANGLE, { x:cx+0.14, y:cy+0.76, w:0.12, h:0.12, fill:{color:C.green}, line:{width:0} });
    sl.addText(pro, { x:cx+0.32, y:cy+0.70, w:cw-0.44, h:0.30, fontSize:10, color:C.textDark, valign:"top", fontFace:"Calibri", margin:0 });
    // Con
    sl.addShape(pres.shapes.RECTANGLE, { x:cx+0.14, y:cy+1.08, w:0.12, h:0.12, fill:{color:C.red}, line:{width:0} });
    sl.addText(con,  { x:cx+0.32, y:cy+1.02, w:cw-0.44, h:0.55, fontSize:10, color:C.textMid, valign:"top", fontFace:"Calibri", margin:0 });
  });

  // Bottom justification banner
  sl.addShape(pres.shapes.RECTANGLE, { x:AW+M, y:H-0.50, w:W-AW-2*M, h:0.42, fill:{color:C.teal}, line:{width:0}, shadow:sh() });
  sl.addText("Literature consensus: gradient boosting achieves best-in-class performance on structured, labeled fraud data — justifying this thesis's model selection.", {
    x:AW+M+0.18, y:H-0.50, w:W-AW-2*M-0.25, h:0.42,
    fontSize:11, bold:true, color:C.white, valign:"middle", fontFace:"Calibri", margin:0
  });
}

// ─────────────────────────────────────────────────────────────
// SLIDE 06 — DATASET: IEEE-CIS
// ─────────────────────────────────────────────────────────────
{
  const sl = pres.addSlide();
  lightSlide(sl, "Dataset: IEEE-CIS Fraud Detection", 6);

  // Left: description
  const lx = AW+M, lw = 5.6, y0 = 1.05;

  card(sl, lx, y0, lw, 3.95);
  sl.addShape(pres.shapes.RECTANGLE, { x:lx, y:y0, w:lw, h:0.45, fill:{color:C.navyDark}, line:{width:0} });
  sl.addText("Source: IEEE-CIS Kaggle Competition", { x:lx+0.15, y:y0, w:lw-0.2, h:0.45, fontSize:12, bold:true, color:C.white, valign:"middle", fontFace:"Calibri", margin:0 });

  const bullets = [
    { h:"Transaction Table",  d:"Contains TransactionDT, TransactionAmt, ProductCD, card types, email domains, and 300+ Vesta-engineered features." },
    { h:"Identity Table",     d:"Device type, browser, OS, network information, and biometric proxy features linked to transactions." },
    { h:"Target Variable",    d:"Binary: isFraud = 1 for fraudulent, 0 for legitimate. Severe imbalance: ~3.5% positive rate." },
    { h:"Why IEEE-CIS?",      d:"Reflects real e-commerce complexity — not a toy dataset. Used in Kaggle competition where top solutions achieved ~96% ROC-AUC." },
  ];

  bullets.forEach(({ h, d }, i) => {
    const by = y0 + 0.62 + i * 0.83;
    sl.addText(h, { x:lx+0.22, y:by,       w:lw-0.35, h:0.30, fontSize:12, bold:true, color:C.textDark, valign:"middle", fontFace:"Calibri", margin:0 });
    sl.addText(d, { x:lx+0.22, y:by+0.30, w:lw-0.35, h:0.45, fontSize:10.5, color:C.textMid, valign:"top", fontFace:"Calibri", margin:0 });
  });

  // Right: metric callouts
  const rx = lx + lw + 0.30, rw = 2.72;
  const metrics = [
    { val:"590,540", lbl:"Total Transactions",     bg:C.navyDark },
    { val:"434",     lbl:"Feature Columns",         bg:C.teal },
    { val:"3.5%",    lbl:"Fraud Rate (Positive)",   bg:C.red },
    { val:"1:5",     lbl:"Downsample Ratio Tested", bg:C.purple },
  ];
  metrics.forEach(({ val, lbl, bg }, i) => {
    metricBox(sl, rx, y0 + i*0.99, rw, 0.88, val, lbl, bg, C.white);
  });

  // Extra note
  sl.addShape(pres.shapes.RECTANGLE, { x:rx, y:y0+4*0.99, w:rw, h:0.52, fill:{color:C.tealPale}, line:{color:C.teal, width:1} });
  sl.addText("Chronological data split\nto simulate real deployment", { x:rx+0.1, y:y0+4*0.99+0.04, w:rw-0.15, h:0.50, fontSize:9.5, color:C.textDark, valign:"middle", fontFace:"Calibri", margin:0 });
}

// ─────────────────────────────────────────────────────────────
// SLIDE 07 — METHODOLOGY PIPELINE
// ─────────────────────────────────────────────────────────────
{
  const sl = pres.addSlide();
  lightSlide(sl, "Methodology Pipeline", 7);

  const steps = [
    { n:1, title:"Preprocessing",       desc:"Handle missingness, encode categorical variables, prepare temporal features", color:C.teal },
    { n:2, title:"Feature Engineering", desc:"Build behavioral aggregates: mean, relative, avg, std, frequency features",  color:C.purple },
    { n:3, title:"Model Training",       desc:"Train XGBoost, LightGBM, CatBoost with time-aware cross-validation",        color:C.navyMid },
    { n:4, title:"Evaluation",           desc:"Compare ROC-AUC, PR-AUC, Precision, Recall, F1 across all configurations",  color:C.green },
    { n:5, title:"Sensitivity Tests",   desc:"Downsampling, reduced features, and decision threshold tuning experiments",  color:C.amber },
  ];

  const x0 = AW+M, stepW = 1.72, stepH = 2.6, gx = 0.12, cy = 1.25;

  steps.forEach(({ n, title, desc, color }, i) => {
    const cx = x0 + i*(stepW+gx);

    // Arrow connector (not after last step)
    if (i < steps.length-1) {
      sl.addText("›", {
        x:cx+stepW, y:cy+0.55, w:gx, h:0.8,
        fontSize:22, bold:true, color:"94A3B8", align:"center", valign:"middle",
        fontFace:"Calibri", margin:0
      });
    }

    // Step card
    card(sl, cx, cy, stepW, stepH);
    // Top color block
    sl.addShape(pres.shapes.RECTANGLE, { x:cx, y:cy, w:stepW, h:1.1, fill:{color}, line:{width:0} });

    // Circle with step number
    const cr = 0.28;
    sl.addShape(pres.shapes.OVAL, { x:cx+stepW/2-cr, y:cy+0.28, w:cr*2, h:cr*2, fill:{color:C.white}, line:{width:0} });
    sl.addText(String(n), { x:cx+stepW/2-cr, y:cy+0.28, w:cr*2, h:cr*2, fontSize:16, bold:true, color, align:"center", valign:"middle", fontFace:"Calibri", margin:0 });

    sl.addText(title, { x:cx+0.1, y:cy+1.18, w:stepW-0.2, h:0.48, fontSize:11.5, bold:true, color:C.textDark, align:"center", valign:"middle", fontFace:"Calibri", margin:0 });
    sl.addText(desc,  { x:cx+0.12,y:cy+1.68, w:stepW-0.24,h:0.84, fontSize:9.8,  color:C.textMid,  align:"center", valign:"top",    fontFace:"Calibri", margin:0 });
  });

  // Bottom note
  sl.addText("Time-series cross-validation was used throughout to prevent data leakage and simulate real-world deployment conditions.", {
    x:AW+M, y:H-0.38, w:W-AW-2*M, h:0.32,
    fontSize:10, color:C.textMuted, align:"center", fontFace:"Calibri", margin:0
  });
}

// ─────────────────────────────────────────────────────────────
// SLIDE 08 — CHRONOLOGICAL DATA SPLIT
// ─────────────────────────────────────────────────────────────
{
  const sl = pres.addSlide();
  lightSlide(sl, "Chronological Data Split & Time-Series Validation", 8);

  // ── Layout: left 6.55" for timeline, right 2.3" for callouts ──
  const tlX = AW+M, tlW = 6.55;           // timeline confined to left column
  const by = 1.28, bh = 0.65;
  const trainW = tlW * 0.72;              // 4.72"
  const valW   = tlW * 0.09;              // 0.59"
  const testW  = tlW * 0.19;              // 1.24"

  // Temporal direction label + arrow line (well below header)
  sl.addText("→  Chronological order (TransactionDT) — no shuffling at any stage", {
    x:tlX, y:0.92, w:tlW, h:0.26,
    fontSize:10, bold:true, color:C.teal, fontFace:"Calibri", margin:0
  });
  sl.addShape(pres.shapes.LINE, { x:tlX, y:1.16, w:tlW, h:0, line:{color:C.teal, width:1} });

  // Train block
  sl.addShape(pres.shapes.RECTANGLE, { x:tlX,                    y:by, w:trainW, h:bh, fill:{color:C.navyMid}, line:{width:0}, shadow:sh() });
  sl.addText("TRAINING SET  —  72%  (chronological first)", {
    x:tlX+0.15, y:by, w:trainW-0.2, h:bh,
    fontSize:11.5, bold:true, color:C.white, valign:"middle", fontFace:"Calibri", margin:0
  });

  // Val block
  sl.addShape(pres.shapes.RECTANGLE, { x:tlX+trainW,             y:by, w:valW,   h:bh, fill:{color:C.teal},  line:{width:0}, shadow:sh() });
  sl.addText("VAL\n9%", {
    x:tlX+trainW+0.03, y:by, w:valW-0.04, h:bh,
    fontSize:8.5, bold:true, color:C.white, align:"center", valign:"middle", fontFace:"Calibri", margin:0
  });

  // Test block
  sl.addShape(pres.shapes.RECTANGLE, { x:tlX+trainW+valW,        y:by, w:testW,  h:bh, fill:{color:C.green}, line:{width:0}, shadow:sh() });
  sl.addText("TEST  —  19%", {
    x:tlX+trainW+valW+0.10, y:by, w:testW-0.14, h:bh,
    fontSize:11, bold:true, color:C.white, valign:"middle", fontFace:"Calibri", margin:0
  });

  // Timeline tick labels
  sl.addShape(pres.shapes.LINE, { x:tlX, y:by+bh+0.06, w:tlW, h:0, line:{color:C.textMuted, width:0.75} });
  sl.addText("Jan 2017",   { x:tlX,               y:by+bh+0.10, w:0.75, h:0.22, fontSize:8.5, color:C.textMuted, fontFace:"Calibri", margin:0 });
  sl.addText("~Oct 2017",  { x:tlX+trainW-0.40,   y:by+bh+0.10, w:0.82, h:0.22, fontSize:8.5, color:C.textMuted, fontFace:"Calibri", margin:0 });
  sl.addText("Dec 2017",   { x:tlX+tlW-0.75,      y:by+bh+0.10, w:0.75, h:0.22, fontSize:8.5, align:"right", color:C.textMuted, fontFace:"Calibri", margin:0 });

  // ── Time-series CV diagram (stays within trainW) ────────────
  const cvY = 2.36, cvX = tlX, foldW = trainW / 5;
  sl.addText("Time-Series Cross-Validation (within training set)", {
    x:cvX, y:cvY, w:trainW+valW+testW, h:0.28,
    fontSize:11, bold:true, color:C.textDark, fontFace:"Calibri", margin:0
  });

  for (let f = 0; f < 5; f++) {
    const fy = cvY + 0.34 + f * 0.36;
    // Prior folds (darker fill = already trained)
    if (f > 0) sl.addShape(pres.shapes.RECTANGLE, { x:cvX, y:fy, w:f*foldW, h:0.28, fill:{color:C.navyMid, transparency:35}, line:{width:0} });
    // Current active fold (teal)
    sl.addShape(pres.shapes.RECTANGLE, { x:cvX+f*foldW, y:fy, w:foldW, h:0.28, fill:{color:C.teal}, line:{width:0} });
    sl.addText(`Fold ${f+1}`, { x:cvX+f*foldW+0.05, y:fy, w:foldW-0.08, h:0.28, fontSize:9, bold:true, color:C.white, valign:"middle", fontFace:"Calibri", margin:0 });
  }

  // ── Key callouts (right column — separated from timeline) ───
  const callouts = [
    { color:C.teal,   t:"No Data Leakage",     d:"Future information never used to train on past data" },
    { color:C.green,  t:"Simulates Deployment", d:"Model validated on transactions it has never seen" },
    { color:C.purple, t:"Robust Evaluation",    d:"5-fold CV provides stable performance estimates" },
  ];
  const kcx = tlX + tlW + 0.22;             // starts after timeline + gap
  const kcw = W - kcx - M;                  // ~2.32"
  const kcy0 = by, kch = 0.88, kcg = 0.14;
  callouts.forEach(({ color, t, d }, i) => {
    const ky = kcy0 + i * (kch + kcg);
    accentCard(sl, kcx, ky, kcw, kch, color, true);
    sl.addText(t, { x:kcx+0.22, y:ky+0.06, w:kcw-0.30, h:0.28, fontSize:10.5, bold:true, color:C.textDark, valign:"middle", fontFace:"Calibri", margin:0 });
    sl.addText(d, { x:kcx+0.22, y:ky+0.36, w:kcw-0.30, h:0.46, fontSize:9.5, color:C.textMid, valign:"top", fontFace:"Calibri", margin:0 });
  });

  // Bottom footnote
  sl.addShape(pres.shapes.RECTANGLE, { x:AW+M, y:H-0.38, w:W-AW-2*M, h:0.32, fill:{color:C.navyDark}, line:{width:0} });
  sl.addText("Chronological split is essential for fraud detection — random splitting would overestimate performance by leaking future fraud patterns into training.", {
    x:AW+M+0.15, y:H-0.38, w:W-AW-2*M-0.20, h:0.32, fontSize:9.5, color:C.tealBright, valign:"middle", fontFace:"Calibri", margin:0
  });
}

// ─────────────────────────────────────────────────────────────
// SLIDE 09 — EDA KEY FINDINGS
// ─────────────────────────────────────────────────────────────
{
  const sl = pres.addSlide();
  lightSlide(sl, "Exploratory Data Analysis: Key Patterns", 9);

  const findings = [
    { n:"01", color:C.red,    title:"Severe Class Imbalance",
      body:"Only ~3.5% of transactions are fraudulent. A naive majority-predictor achieves 96.5% accuracy — making recall-oriented metrics essential for meaningful evaluation." },
    { n:"02", color:C.amber,  title:"Temporal Fraud Signatures",
      body:"Fraud rates vary significantly by hour of day and day of week. Time-series patterns reveal that fraudsters tend to operate at specific temporal windows." },
    { n:"03", color:C.teal,   title:"Widespread Missing Values",
      body:"Missing values are unevenly distributed across feature groups (identity table has most). Handling missingness is central to model performance, not just a preprocessing detail." },
    { n:"04", color:C.purple, title:"Behavioral & Categorical Signals",
      body:"Transaction amount, decimal patterns (3 decimal places = higher fraud risk), product category, and card/email domain features all show meaningful shifts between fraud and non-fraud." },
  ];

  const x0 = AW+M, y0 = 1.0, cw = 4.38, ch = 1.78, gx = 0.22, gy = 0.18;
  findings.forEach(({ n, color, title, body }, i) => {
    const cx = x0 + (i%2)*(cw+gx);
    const cy = y0 + Math.floor(i/2)*(ch+gy);
    accentCard(sl, cx, cy, cw, ch, color);
    // Number callout
    sl.addShape(pres.shapes.OVAL, { x:cx+0.12, y:cy+0.15, w:0.46, h:0.46, fill:{color}, line:{width:0} });
    sl.addText(n, { x:cx+0.12, y:cy+0.15, w:0.46, h:0.46, fontSize:11, bold:true, color:C.white, align:"center", valign:"middle", fontFace:"Calibri", margin:0 });
    sl.addText(title, { x:cx+0.70, y:cy+0.15, w:cw-0.82, h:0.46, fontSize:12.5, bold:true, color:C.textDark, valign:"middle", fontFace:"Calibri", margin:0 });
    sl.addText(body,  { x:cx+0.12, y:cy+0.68, w:cw-0.22, h:1.02, fontSize:10.5, color:C.textMid, valign:"top", fontFace:"Calibri", margin:0 });
  });

  // Takeaway banner
  sl.addShape(pres.shapes.RECTANGLE, { x:AW+M, y:4.87, w:W-AW-2*M, h:0.40, fill:{color:C.navyDark}, line:{width:0} });
  sl.addText("EDA TAKEAWAY: Fraud is not random noise — it leaves temporal, behavioral, and categorical signatures that machine learning can exploit.", {
    x:AW+M+0.18, y:4.87, w:W-AW-2*M-0.2, h:0.40,
    fontSize:10, bold:false, color:C.tealBright, valign:"middle", fontFace:"Calibri", margin:0
  });
}

// ─────────────────────────────────────────────────────────────
// SLIDE 10 — FEATURE ENGINEERING STRATEGY
// ─────────────────────────────────────────────────────────────
{
  const sl = pres.addSlide();
  lightSlide(sl, "Feature Engineering Strategy", 10);

  const cats = [
    { color:C.teal,   title:"Behavioral Aggregates",
      items:["User-level transaction means (_mean)", "Relative deviation from user baseline (_rel)", "Activity frequency signals (_freq, _std, _avg)"] },
    { color:C.purple, title:"Temporal Patterns",
      items:["Hour of day & day of week context", "Delta-time between transactions", "Sequence-aware user behavior signals"] },
    { color:C.green,  title:"Entity Interactions",
      items:["Amount × card type features", "Amount × product category signals", "Cross-feature transaction relationships"] },
    { color:C.amber,  title:"Why It Matters",
      items:["Captures hidden transaction habits", "Improves discrimination beyond raw fields", "Engineered features ranked top predictors across all 3 models"] },
  ];

  const x0 = AW+M, y0 = 1.05, cw = 4.38, ch = 2.25, gx = 0.22, gy = 0.22;
  cats.forEach(({ color, title, items }, i) => {
    const cx = x0 + (i%2)*(cw+gx);
    const cy = y0 + Math.floor(i/2)*(ch+gy);
    card(sl, cx, cy, cw, ch);
    sl.addShape(pres.shapes.RECTANGLE, { x:cx, y:cy, w:cw, h:0.42, fill:{color}, line:{width:0} });
    sl.addText(title, { x:cx+0.14, y:cy, w:cw-0.2, h:0.42, fontSize:12.5, bold:true, color:C.white, valign:"middle", fontFace:"Calibri", margin:0 });
    items.forEach((item, j) => {
      const iy = cy + 0.55 + j*0.55;
      sl.addShape(pres.shapes.RECTANGLE, { x:cx+0.17, y:iy+0.12, w:0.14, h:0.14, fill:{color}, line:{width:0} });
      sl.addText(item, { x:cx+0.38, y:iy, w:cw-0.52, h:0.42, fontSize:10.8, color:C.textMid, valign:"middle", fontFace:"Calibri", margin:0 });
    });
  });
}

// ─────────────────────────────────────────────────────────────
// SLIDE 11 — SHAP FEATURE IMPORTANCE & EXPLAINABILITY
// ─────────────────────────────────────────────────────────────
{
  const sl = pres.addSlide();
  lightSlide(sl, "SHAP: Feature Importance & Model Explainability", 11);

  // ── Left: horizontal feature importance bars ────────────────
  const lx = AW+M, ly0 = 1.05, lw = 5.45;

  card(sl, lx, ly0, lw, 4.18);
  sl.addShape(pres.shapes.RECTANGLE, { x:lx, y:ly0, w:lw, h:0.42, fill:{color:C.navyDark}, line:{width:0} });
  sl.addText("Top Predictors by SHAP Importance (Cross-Model)", {
    x:lx+0.15, y:ly0, w:lw-0.20, h:0.42,
    fontSize:11.5, bold:true, color:C.white, valign:"middle", fontFace:"Calibri", margin:0
  });

  // Feature bars — names embedded INSIDE the bar as white bold text
  const features = [
    { name:"card1  (card identifier)",         val:0.92, color:C.teal },
    { name:"TransactionAmt_mean  (behavioral)", val:0.87, color:C.teal },
    { name:"TransactionAmt",                    val:0.80, color:C.tealBright },
    { name:"C1  (transaction count)",          val:0.74, color:C.navyMid },
    { name:"D1  (time delta)",                  val:0.68, color:C.navyMid },
    { name:"addr1  (billing address)",          val:0.62, color:C.navyMid },
    { name:"card1_addr1  (interaction)",        val:0.55, color:C.purple },
  ];

  const barX = lx + 0.18, maxBarW = lw - 0.38, barH = 0.42, barGap = 0.52;
  features.forEach(({ name, val, color }, i) => {
    const fy = ly0 + 0.58 + i * barGap;
    // Background track (full width, light)
    sl.addShape(pres.shapes.RECTANGLE, { x:barX, y:fy, w:maxBarW, h:barH, fill:{color:"DDE8F4"}, line:{width:0} });
    // Filled colored bar
    sl.addShape(pres.shapes.RECTANGLE, { x:barX, y:fy, w:maxBarW*val, h:barH, fill:{color}, line:{width:0} });
    // Rank number badge (slightly darker overlay on left of bar)
    sl.addShape(pres.shapes.RECTANGLE, { x:barX, y:fy, w:0.34, h:barH, fill:{color, transparency:20}, line:{width:0} });
    sl.addText(String(i+1), { x:barX, y:fy, w:0.34, h:barH, fontSize:12, bold:true, color:C.white, align:"center", valign:"middle", fontFace:"Calibri", margin:0 });
    // Feature name inside bar (white bold)
    sl.addText(name, { x:barX+0.38, y:fy, w:maxBarW*val-0.44, h:barH, fontSize:10.5, bold:true, color:C.white, valign:"middle", fontFace:"Calibri", margin:0 });
    // Percentage label on track background after bar end
    sl.addText(`${Math.round(val*100)}%`, {
      x:barX + maxBarW*val + 0.06, y:fy, w:0.40, h:barH,
      fontSize:10.5, bold:true, color:C.textDark, valign:"middle", fontFace:"Calibri", margin:0
    });
  });

  // ── Right: What SHAP provides ───────────────────────────────
  const rx = lx + lw + 0.28, rw = 3.52, ry0 = ly0;

  // Global explainability box
  card(sl, rx, ry0, rw, 1.92, C.bgCard);
  sl.addShape(pres.shapes.RECTANGLE, { x:rx, y:ry0, w:rw, h:0.42, fill:{color:C.teal}, line:{width:0} });
  sl.addText("Global Explainability", { x:rx+0.14, y:ry0, w:rw-0.20, h:0.42, fontSize:12, bold:true, color:C.white, valign:"middle", fontFace:"Calibri", margin:0 });
  sl.addText("SHAP ranks the features driving fraud detection across all transactions — showing which variables the model relies on most. Enables audit, compliance reporting, and feature selection.", {
    x:rx+0.14, y:ry0+0.48, w:rw-0.22, h:1.36,
    fontSize:10.2, color:C.textMid, valign:"top", fontFace:"Calibri", margin:0
  });

  // Local explainability box
  card(sl, rx, ry0+2.02, rw, 1.92, C.bgCard);
  sl.addShape(pres.shapes.RECTANGLE, { x:rx, y:ry0+2.02, w:rw, h:0.42, fill:{color:C.purple}, line:{width:0} });
  sl.addText("Local Explainability", { x:rx+0.14, y:ry0+2.02, w:rw-0.20, h:0.42, fontSize:12, bold:true, color:C.white, valign:"middle", fontFace:"Calibri", margin:0 });
  sl.addText("SHAP assigns a contribution score to each feature for every individual prediction — answering \"why was this specific transaction flagged?\" This enables analyst review and appeals processes.", {
    x:rx+0.14, y:ry0+2.02+0.48, w:rw-0.22, h:1.36,
    fontSize:10.2, color:C.textMid, valign:"top", fontFace:"Calibri", margin:0
  });

  // Banner: "Already delivered in this thesis"
  sl.addShape(pres.shapes.RECTANGLE, { x:AW+M, y:H-0.40, w:W-AW-2*M, h:0.34, fill:{color:C.teal}, line:{width:0} });
  sl.addText("SHAP explainability is already delivered in this thesis — both global rankings and per-prediction attribution scores were computed for all three models.", {
    x:AW+M+0.15, y:H-0.40, w:W-AW-2*M-0.20, h:0.34, fontSize:10, bold:true, color:C.white, valign:"middle", fontFace:"Calibri", margin:0
  });
}

// ─────────────────────────────────────────────────────────────
// SLIDE 12 — EXPERIMENTAL SETUP
// ─────────────────────────────────────────────────────────────
{
  const sl = pres.addSlide();
  lightSlide(sl, "Experimental Configurations", 12);

  // Left: 4 configurations
  const configs = [
    { n:"C1", color:C.teal,   title:"Baseline — Full Features, Imbalanced Data",
      desc:"Train on original data without any rebalancing. Establishes the performance ceiling for raw gradient boosting." },
    { n:"C2", color:C.purple, title:"Majority-Class Downsampling (1:5 Ratio)",
      desc:"Reduce legitimate transactions to 1:5 ratio. Tests whether controlled rebalancing helps models learn fraud patterns better." },
    { n:"C3", color:C.green,  title:"Reduced Feature Set (SHAP-Based Selection)",
      desc:"Select features based on cross-model SHAP importance agreement (top 30% per model, ≥2 models → 215 features). Tests competitive compact models." },
    { n:"C4", color:C.amber,  title:"Decision Threshold Tuning (0.1)",
      desc:"Lower threshold from default 0.5 to 0.1. Demonstrates the operational precision–recall trade-off." },
  ];

  const cx0 = AW+M, cw = 5.8, ch = 0.92, cy0 = 1.05, cg = 0.18;
  configs.forEach(({ n, color, title, desc }, i) => {
    const y = cy0 + i*(ch+cg);
    accentCard(sl, cx0, y, cw, ch, color);
    sl.addShape(pres.shapes.RECTANGLE, { x:cx0, y, w:0.50, h:ch, fill:{color}, line:{width:0} });
    sl.addText(n, { x:cx0, y, w:0.50, h:ch, fontSize:14, bold:true, color:C.white, align:"center", valign:"middle", fontFace:"Calibri", margin:0 });
    sl.addText(title, { x:cx0+0.60, y:y+0.06, w:cw-0.70, h:0.36, fontSize:11.5, bold:true, color:C.textDark, valign:"middle", fontFace:"Calibri", margin:0 });
    sl.addText(desc,  { x:cx0+0.60, y:y+0.44, w:cw-0.70, h:0.40, fontSize:10,   color:C.textMid,  valign:"top",    fontFace:"Calibri", margin:0 });
  });

  // Right: models + primary metric
  const rx = cx0+cw+0.35, rw = 3.0;
  card(sl, rx, 1.05, rw, 2.0);
  sl.addShape(pres.shapes.RECTANGLE, { x:rx, y:1.05, w:rw, h:0.42, fill:{color:C.navyDark}, line:{width:0} });
  sl.addText("MODELS COMPARED", { x:rx+0.1, y:1.05, w:rw-0.15, h:0.42, fontSize:11, bold:true, color:C.white, valign:"middle", fontFace:"Calibri", margin:0 });
  const models = [
    { name:"XGBoost",  color:C.amber },
    { name:"LightGBM", color:C.teal  },
    { name:"CatBoost", color:C.purple},
  ];
  models.forEach(({ name, color }, i) => {
    const my = 1.60 + i * 0.44;
    sl.addShape(pres.shapes.OVAL, { x:rx+0.22, y:my, w:0.22, h:0.22, fill:{color}, line:{width:0} });
    sl.addText(name, { x:rx+0.52, y:my-0.04, w:rw-0.65, h:0.30, fontSize:12.5, bold:true, color:C.textDark, valign:"middle", fontFace:"Calibri", margin:0 });
  });

  card(sl, rx, 3.20, rw, 1.8);
  sl.addShape(pres.shapes.RECTANGLE, { x:rx, y:3.20, w:rw, h:0.42, fill:{color:C.navyDark}, line:{width:0} });
  sl.addText("EVALUATION METRICS", { x:rx+0.1, y:3.20, w:rw-0.15, h:0.42, fontSize:11, bold:true, color:C.white, valign:"middle", fontFace:"Calibri", margin:0 });
  const metrics2 = ["ROC-AUC (primary)", "PR-AUC", "Precision / Recall", "F1-Score"];
  metrics2.forEach((m, i) => {
    const my = 3.75 + i*0.29;
    sl.addShape(pres.shapes.RECTANGLE, { x:rx+0.22, y:my+0.06, w:0.16, h:0.16, fill:{color:C.teal}, line:{width:0} });
    sl.addText(m, { x:rx+0.46, y:my, w:rw-0.58, h:0.29, fontSize:11, color:C.textMid, valign:"middle", fontFace:"Calibri", margin:0 });
  });
}

// ─────────────────────────────────────────────────────────────
// SLIDE 13 — RESULTS: BASELINE (Full Features, Imbalanced)
// ─────────────────────────────────────────────────────────────
{
  const sl = pres.addSlide();
  lightSlide(sl, "Results: Baseline — Full Features, Imbalanced Data", 13);

  // Headline callout
  sl.addShape(pres.shapes.RECTANGLE, { x:AW+M, y:1.03, w:W-AW-2*M, h:0.52, fill:{color:C.teal}, line:{width:0}, shadow:sh() });
  sl.addText("LightGBM achieves the strongest baseline ROC-AUC at 0.918, ahead of CatBoost (0.910) and XGBoost (0.896)", {
    x:AW+M+0.18, y:1.03, w:W-AW-2*M-0.25, h:0.52,
    fontSize:12.5, bold:true, color:C.white, valign:"middle", fontFace:"Calibri", margin:0
  });

  // Bar chart
  sl.addChart(pres.charts.BAR, [{
    name:"ROC-AUC",
    labels:["XGBoost", "LightGBM", "CatBoost"],
    values:[0.896, 0.918, 0.910]
  }], {
    x:AW+M, y:1.68, w:5.3, h:3.45,
    barDir:"col",
    chartColors:[C.amber, C.teal, C.purple],
    chartArea:{ fill:{color:C.bgCard}, roundedCorners:false },
    catAxisLabelColor:"475569",
    valAxisLabelColor:"475569",
    valAxisMinVal:0.87, valAxisMaxVal:0.93,
    valGridLine:{ color:"E2E8F0", size:0.5 },
    catGridLine:{ style:"none" },
    showLegend:false,
    shadow: sh(),
  });

  // Right panel: key metrics
  const rx = 6.05, rw = 3.60, ry0 = 1.68;
  const rows = [
    { model:"XGBoost",  mc:C.amber,  auc:"0.896", prauc:"0.496", f1:"0.384", prec:"0.269", rec:"0.671" },
    { model:"LightGBM", mc:C.teal,   auc:"0.918", prauc:"0.537", f1:"0.506", prec:"0.588", rec:"0.456" },
    { model:"CatBoost", mc:C.purple, auc:"0.910", prauc:"0.510", f1:"0.475", prec:"0.490", rec:"0.461" },
  ];

  // Table header
  card(sl, rx, ry0, rw, 0.42, C.navyDark, false);
  sl.addShape(pres.shapes.RECTANGLE, { x:rx, y:ry0, w:rw, h:0.42, fill:{color:C.navyDark}, line:{width:0} });
  const hds = ["Model","AUC","PR-AUC","F1","Prec.","Rec."];
  const colWs = [1.05, 0.48, 0.52, 0.42, 0.48, 0.48];
  let hx = rx;
  hds.forEach((h, hi) => {
    sl.addText(h, { x:hx+0.03, y:ry0+0.02, w:colWs[hi]-0.05, h:0.38, fontSize:9.5, bold:true, color:C.white, align:"center", valign:"middle", fontFace:"Calibri", margin:0 });
    hx += colWs[hi];
  });

  rows.forEach(({ model, mc, auc, prauc, f1, prec, rec }, ri) => {
    const ry = ry0 + 0.42 + ri*0.57;
    sl.addShape(pres.shapes.RECTANGLE, { x:rx, y:ry, w:rw, h:0.57, fill:{color: ri%2===0?C.bgCard:C.bgCardAlt}, line:{color:"E2E8F0", width:1} });
    sl.addShape(pres.shapes.RECTANGLE, { x:rx, y:ry, w:0.06, h:0.57, fill:{color:mc}, line:{width:0} });
    const vals = [model, auc, prauc, f1, prec, rec];
    let dx = rx;
    vals.forEach((v, vi) => {
      sl.addText(v, { x:dx+0.06, y:ry+0.04, w:colWs[vi]-0.08, h:0.49, fontSize:10.5, bold:vi===0||vi===1, color:vi===0?mc:C.textDark, align:"center", valign:"middle", fontFace:"Calibri", margin:0 });
      dx += colWs[vi];
    });
  });

  // Insight note
  sl.addShape(pres.shapes.RECTANGLE, { x:rx, y:ry0+0.42+3*0.57+0.1, w:rw, h:1.18, fill:{color:C.tealPale}, line:{color:C.teal, width:1} });
  sl.addText([
    { text:"XGBoost", options:{bold:true} },
    { text:" shows highest recall (0.671) — catching more fraud — but with much lower precision (0.269), generating more false alarms.\n\n" },
    { text:"LightGBM", options:{bold:true} },
    { text:" achieves the best overall balance: highest AUC, PR-AUC, and F1." }
  ], { x:rx+0.12, y:ry0+0.42+3*0.57+0.16, w:rw-0.20, h:1.06, fontSize:10, color:C.textDark, valign:"top", fontFace:"Calibri", margin:0 });
}

// ─────────────────────────────────────────────────────────────
// SLIDE 14 — RESULTS: DOWNSAMPLING
// ─────────────────────────────────────────────────────────────
{
  const sl = pres.addSlide();
  lightSlide(sl, "Results: Majority-Class Downsampling (1:5)", 14);

  sl.addShape(pres.shapes.RECTANGLE, { x:AW+M, y:1.03, w:W-AW-2*M, h:0.52, fill:{color:C.purple}, line:{width:0}, shadow:sh() });
  sl.addText("Downsampling improves fraud sensitivity across all models without significantly collapsing ROC-AUC performance", {
    x:AW+M+0.18, y:1.03, w:W-AW-2*M-0.25, h:0.52,
    fontSize:12.5, bold:true, color:C.white, valign:"middle", fontFace:"Calibri", margin:0
  });

  // Recall comparison: before vs after
  const bx = AW+M, bw = 4.3, ry0 = 1.75;

  // Before panel
  card(sl, bx, ry0, bw, 3.4);
  sl.addShape(pres.shapes.RECTANGLE, { x:bx, y:ry0, w:bw, h:0.42, fill:{color:C.textMid}, line:{width:0} });
  sl.addText("BEFORE: Imbalanced", { x:bx+0.1, y:ry0, w:bw-0.15, h:0.42, fontSize:12, bold:true, color:C.white, valign:"middle", fontFace:"Calibri", margin:0 });

  const before = [
    { model:"XGBoost",  mc:C.amber,  recall:"0.671", auc:"0.896" },
    { model:"LightGBM", mc:C.teal,   recall:"0.456", auc:"0.918" },
    { model:"CatBoost", mc:C.purple, recall:"0.461", auc:"0.910" },
  ];
  before.forEach(({ model, mc, recall, auc }, i) => {
    const y = ry0 + 0.58 + i*0.88;
    sl.addShape(pres.shapes.RECTANGLE, { x:bx+0.18, y, w:bw-0.35, h:0.75, fill:{color:C.bgCardAlt}, line:{color:"E2E8F0", width:1} });
    sl.addShape(pres.shapes.RECTANGLE, { x:bx+0.18, y, w:0.06, h:0.75, fill:{color:mc}, line:{width:0} });
    sl.addText(model,  { x:bx+0.30, y:y+0.06, w:1.4, h:0.30, fontSize:12, bold:true, color:mc, valign:"middle", fontFace:"Calibri", margin:0 });
    sl.addText(`Recall: ${recall}  |  ROC-AUC: ${auc}`, { x:bx+0.30, y:y+0.38, w:bw-0.55, h:0.28, fontSize:10.5, color:C.textMid, valign:"top", fontFace:"Calibri", margin:0 });
  });

  // Arrow in middle — clean text only, no rectangle artifact
  sl.addText("→", { x:bx+bw+0.05, y:2.55, w:0.65, h:0.70, fontSize:34, bold:true, color:C.teal, align:"center", valign:"middle", fontFace:"Calibri", margin:0 });
  sl.addText("1:5\nDS", { x:bx+bw+0.05, y:3.28, w:0.65, h:0.44, fontSize:9.5, color:C.textMuted, align:"center", fontFace:"Calibri", margin:0 });

  // After panel
  const ax = bx+bw+0.70, aw = 4.3;
  card(sl, ax, ry0, aw, 3.4);
  sl.addShape(pres.shapes.RECTANGLE, { x:ax, y:ry0, w:aw, h:0.42, fill:{color:C.green}, line:{width:0} });
  sl.addText("AFTER: 1:5 Downsampled", { x:ax+0.1, y:ry0, w:aw-0.15, h:0.42, fontSize:12, bold:true, color:C.white, valign:"middle", fontFace:"Calibri", margin:0 });

  const after = [
    { model:"XGBoost",  mc:C.amber,  recall:"0.703", delta:"+4.8%",  auc:"0.908" },
    { model:"LightGBM", mc:C.teal,   recall:"0.626", delta:"+37.3%", auc:"0.917" },
    { model:"CatBoost", mc:C.purple, recall:"0.757", delta:"+64.2%", auc:"0.916" },
  ];
  after.forEach(({ model, mc, recall, delta, auc }, i) => {
    const y = ry0 + 0.58 + i*0.88;
    sl.addShape(pres.shapes.RECTANGLE, { x:ax+0.18, y, w:aw-0.35, h:0.75, fill:{color:C.bgCardAlt}, line:{color:"E2E8F0", width:1} });
    sl.addShape(pres.shapes.RECTANGLE, { x:ax+0.18, y, w:0.06, h:0.75, fill:{color:mc}, line:{width:0} });
    sl.addText(model,  { x:ax+0.30, y:y+0.06, w:1.4, h:0.30, fontSize:12, bold:true, color:mc, valign:"middle", fontFace:"Calibri", margin:0 });
    sl.addText(`Recall: ${recall}`, { x:ax+0.30, y:y+0.38, w:1.5, h:0.28, fontSize:10.5, color:C.textMid, valign:"top", fontFace:"Calibri", margin:0 });
    sl.addShape(pres.shapes.RECTANGLE, { x:ax+1.88, y:y+0.32, w:0.90, h:0.32, fill:{color:C.greenPale}, line:{width:0} });
    sl.addText(`↑ ${delta}`, { x:ax+1.88, y:y+0.32, w:0.90, h:0.32, fontSize:10, bold:true, color:C.green, align:"center", valign:"middle", fontFace:"Calibri", margin:0 });
    sl.addText(`ROC-AUC: ${auc}`, { x:ax+2.90, y:y+0.38, w:1.0, h:0.28, fontSize:10.5, color:C.textMid, valign:"top", fontFace:"Calibri", margin:0 });
  });

  // Insight
  sl.addShape(pres.shapes.RECTANGLE, { x:AW+M, y:H-0.40, w:W-AW-2*M, h:0.34, fill:{color:C.navyDark}, line:{width:0} });
  sl.addText("Operational reading: controlled rebalancing substantially increases fraud capture with only marginal ROC-AUC cost.", {
    x:AW+M+0.15, y:H-0.40, w:W-AW-2*M-0.20, h:0.34, fontSize:10, color:C.tealBright, valign:"middle", fontFace:"Calibri", margin:0
  });
}

// ─────────────────────────────────────────────────────────────
// SLIDE 15 — RESULTS: FEATURE REDUCTION
// ─────────────────────────────────────────────────────────────
{
  const sl = pres.addSlide();
  lightSlide(sl, "Results: Feature Set Reduction", 15);

  sl.addShape(pres.shapes.RECTANGLE, { x:AW+M, y:1.03, w:W-AW-2*M, h:0.52, fill:{color:C.green}, line:{width:0}, shadow:sh() });
  sl.addText("Compact models remain highly competitive — LightGBM slightly improves to 0.919 ROC-AUC with reduced features", {
    x:AW+M+0.18, y:1.03, w:W-AW-2*M-0.25, h:0.52,
    fontSize:12.5, bold:true, color:C.white, valign:"middle", fontFace:"Calibri", margin:0
  });

  // Left: ROC-AUC comparison chart
  sl.addChart(pres.charts.BAR, [
    { name:"Full Features",    labels:["XGBoost","LightGBM","CatBoost"], values:[0.896, 0.918, 0.910] },
    { name:"Reduced Features", labels:["XGBoost","LightGBM","CatBoost"], values:[0.905, 0.919, 0.914] },
  ], {
    x:AW+M, y:1.72, w:5.4, h:3.32,
    barDir:"col",
    barGrouping:"clustered",
    chartColors:[C.textMuted, C.teal],
    chartArea:{ fill:{color:C.bgCard}, roundedCorners:false },
    catAxisLabelColor:"475569",
    valAxisLabelColor:"475569",
    valAxisMinVal:0.87, valAxisMaxVal:0.93,
    valGridLine:{ color:"E2E8F0", size:0.5 },
    catGridLine:{ style:"none" },
    showLegend:true, legendPos:"b",
    shadow: sh(),
  });

  // Right: insights — ih increased to 0.88 to prevent text overflow
  const rx = 6.12, rw = 3.52;
  const insights = [
    { color:C.green,  title:"Performance Maintained",  desc:"All three models retain near-identical ROC-AUC after compression. LightGBM marginally improves to 0.919." },
    { color:C.teal,   title:"Top Engineered Features", desc:"Behavioral aggregates (_mean, _rel, _avg, _std, _freq) consistently rank as the strongest predictors." },
    { color:C.purple, title:"SHAP-Driven Selection",   desc:"Top 30% importance per model, kept when ≥2 models agree — reduces feature set from 434 to 215." },
    { color:C.amber,  title:"Deployment Advantage",    desc:"Fewer features = faster inference, simpler pipelines, lower maintenance cost in production." },
  ];
  const ry0 = 1.72, ih = 0.86, ig = 0.10;
  insights.forEach(({ color, title, desc }, i) => {
    const iy = ry0 + i*(ih+ig);
    accentCard(sl, rx, iy, rw, ih, color);
    sl.addText(title, { x:rx+0.24, y:iy+0.06, w:rw-0.32, h:0.28, fontSize:11.5, bold:true, color:C.textDark, valign:"middle", fontFace:"Calibri", margin:0 });
    sl.addText(desc,  { x:rx+0.24, y:iy+0.36, w:rw-0.32, h:0.44, fontSize:10,   color:C.textMid,  valign:"top",    fontFace:"Calibri", margin:0 });
  });
}

// ─────────────────────────────────────────────────────────────
// SLIDE 16 — RESULTS: THRESHOLD TUNING
// ─────────────────────────────────────────────────────────────
{
  const sl = pres.addSlide();
  lightSlide(sl, "Results: Decision Threshold Tuning", 16);

  sl.addShape(pres.shapes.RECTANGLE, { x:AW+M, y:1.03, w:W-AW-2*M, h:0.52, fill:{color:C.amber}, line:{width:0}, shadow:sh() });
  sl.addText("Lowering the threshold to 0.1 dramatically increases fraud capture — at the cost of precision. This is a business decision.", {
    x:AW+M+0.18, y:1.03, w:W-AW-2*M-0.25, h:0.52,
    fontSize:12.5, bold:true, color:C.white, valign:"middle", fontFace:"Calibri", margin:0
  });

  // Three model panels in a row
  const models = [
    { name:"XGBoost",  mc:C.amber,
      before:{ rec:"0.671", prec:"0.269" }, after:{ rec:"0.922", prec:"0.085" } },
    { name:"LightGBM", mc:C.teal,
      before:{ rec:"0.456", prec:"0.588" }, after:{ rec:"0.895", prec:"0.110" } },
    { name:"CatBoost", mc:C.purple,
      before:{ rec:"0.461", prec:"0.490" }, after:{ rec:"0.868", prec:"0.108" } },
  ];

  const px0 = AW+M, pw = 2.90, ph = 3.20, pg = 0.25, py0 = 1.73;
  models.forEach(({ name, mc, before, after }, i) => {
    const px = px0 + i*(pw+pg);
    card(sl, px, py0, pw, ph);

    // Model header
    sl.addShape(pres.shapes.RECTANGLE, { x:px, y:py0, w:pw, h:0.44, fill:{color:mc}, line:{width:0} });
    sl.addText(name, { x:px+0.1, y:py0, w:pw-0.15, h:0.44, fontSize:14, bold:true, color:C.white, align:"center", valign:"middle", fontFace:"Calibri", margin:0 });

    // Threshold 0.5 (before)
    sl.addShape(pres.shapes.RECTANGLE, { x:px+0.15, y:py0+0.56, w:pw-0.30, h:0.88, fill:{color:C.bgCardAlt}, line:{color:"E2E8F0", width:1} });
    sl.addText("Threshold 0.5", { x:px+0.22, y:py0+0.60, w:pw-0.40, h:0.28, fontSize:10, bold:true, color:C.textMuted, valign:"middle", fontFace:"Calibri", margin:0 });
    sl.addText(`Recall:    ${before.rec}`, { x:px+0.22, y:py0+0.88, w:pw-0.40, h:0.24, fontSize:11, color:C.textMid, valign:"middle", fontFace:"Calibri", margin:0 });
    sl.addText(`Precision: ${before.prec}`, { x:px+0.22, y:py0+1.10, w:pw-0.40, h:0.24, fontSize:11, color:C.textMid, valign:"middle", fontFace:"Calibri", margin:0 });

    // Down arrow
    sl.addText("↓", { x:px, y:py0+1.52, w:pw, h:0.32, fontSize:20, bold:true, color:C.amber, align:"center", valign:"middle", fontFace:"Calibri", margin:0 });
    sl.addText("threshold → 0.1", { x:px, y:py0+1.80, w:pw, h:0.24, fontSize:9, color:C.textMuted, align:"center", fontFace:"Calibri", margin:0 });

    // Threshold 0.1 (after)
    sl.addShape(pres.shapes.RECTANGLE, { x:px+0.15, y:py0+2.10, w:pw-0.30, h:0.88, fill:{color:C.amberPale}, line:{color:C.amber, width:1} });
    sl.addText("Threshold 0.1", { x:px+0.22, y:py0+2.14, w:pw-0.40, h:0.28, fontSize:10, bold:true, color:C.amber, valign:"middle", fontFace:"Calibri", margin:0 });
    sl.addText(`Recall:    ${after.rec}`, { x:px+0.22, y:py0+2.42, w:pw-0.40, h:0.24, fontSize:11, bold:true, color:C.green, valign:"middle", fontFace:"Calibri", margin:0 });
    sl.addText(`Precision: ${after.prec}`, { x:px+0.22, y:py0+2.64, w:pw-0.40, h:0.24, fontSize:11, bold:true, color:C.red, valign:"middle", fontFace:"Calibri", margin:0 });
  });

  // Bottom callout
  sl.addShape(pres.shapes.RECTANGLE, { x:AW+M, y:H-0.40, w:W-AW-2*M, h:0.34, fill:{color:C.navyDark}, line:{width:0} });
  sl.addText("Institutions should calibrate thresholds against their own fraud-loss tolerance and analyst review capacity.", {
    x:AW+M+0.15, y:H-0.40, w:W-AW-2*M-0.20, h:0.34, fontSize:10, color:C.tealBright, valign:"middle", fontFace:"Calibri", margin:0
  });
}

// ─────────────────────────────────────────────────────────────
// SLIDE 17 — CHAMPION SCORECARD
// ─────────────────────────────────────────────────────────────
{
  const sl = pres.addSlide();
  darkBg(sl);

  sl.addText("CHAMPION SCORECARD", {
    x:0.5, y:0.22, w:9.0, h:0.40,
    fontSize:10, bold:true, color:C.teal, align:"center", charSpacing:5, fontFace:"Calibri", margin:0
  });
  sl.addText("ROC-AUC Across All Configurations", {
    x:0.5, y:0.60, w:9.0, h:0.62,
    fontSize:28, bold:true, color:C.white, align:"center", valign:"middle", fontFace:"Calibri", margin:0
  });

  // Table
  const tx = 1.0, tw = 8.0, ty = 1.45;
  const cols = ["Model", "Baseline\n(Imbalanced)", "Config 2\n(Downsampled)", "Config 3\n(Reduced\nFeatures)"];
  const colW = [1.8, 2.0, 2.0, 2.2];
  const rowH = [0.60, 0.80, 0.80, 0.80];

  // Header row
  let hx = tx;
  cols.forEach((c, ci) => {
    sl.addShape(pres.shapes.RECTANGLE, { x:hx, y:ty, w:colW[ci], h:rowH[0], fill:{color:ci===0?C.navyMid:C.teal}, line:{color:C.navyMid, width:1} });
    sl.addText(c, { x:hx+0.05, y:ty, w:colW[ci]-0.10, h:rowH[0], fontSize:11, bold:true, color:C.white, align:"center", valign:"middle", fontFace:"Calibri", margin:0 });
    hx += colW[ci];
  });

  // Data rows
  const tableData = [
    { model:"LightGBM", mc:C.teal,   vals:["0.918", "0.917", "0.919"], best:[false,false,true] },
    { model:"CatBoost", mc:C.purple, vals:["0.910", "0.916", "0.914"], best:[false,false,false] },
    { model:"XGBoost",  mc:C.amber,  vals:["0.896", "0.908", "0.905"], best:[false,false,false] },
  ];

  tableData.forEach(({ model, mc, vals, best }, ri) => {
    const ry = ty + rowH[0] + ri*rowH[1];
    // Model cell
    sl.addShape(pres.shapes.RECTANGLE, { x:tx, y:ry, w:colW[0], h:rowH[1], fill:{color:C.navy}, line:{color:C.navyMid, width:1} });
    sl.addShape(pres.shapes.RECTANGLE, { x:tx, y:ry, w:0.06, h:rowH[1], fill:{color:mc}, line:{width:0} });
    sl.addText(model, { x:tx+0.12, y:ry, w:colW[0]-0.15, h:rowH[1], fontSize:14, bold:true, color:mc, valign:"middle", fontFace:"Calibri", margin:0 });

    vals.forEach((v, vi) => {
      const cx = tx + colW[0] + vi*colW[vi+1];
      const isHighlight = ri===0; // LightGBM row highlight
      const isBest = best[vi];
      const bgColor = isBest ? C.teal : (isHighlight ? C.navyMid : C.navy);
      sl.addShape(pres.shapes.RECTANGLE, { x:cx, y:ry, w:colW[vi+1], h:rowH[1], fill:{color:bgColor}, line:{color:C.navyMid, width:1} });
      sl.addText(v, { x:cx, y:ry, w:colW[vi+1], h:rowH[1], fontSize:22, bold:true, color:isHighlight?C.white:C.tealPale, align:"center", valign:"middle", fontFace:"Calibri", margin:0 });
    });
  });

  // Bottom note
  sl.addShape(pres.shapes.RECTANGLE, { x:tx, y:ty+rowH[0]+3*rowH[1]+0.15, w:tw, h:0.50, fill:{color:C.navyMid}, line:{color:C.teal, width:1} });
  sl.addText("LightGBM provides the most stable and dominant discriminatory performance across all configurations. Notably, feature reduction (Config 3) slightly improves LightGBM to 0.919.", {
    x:tx+0.15, y:ty+rowH[0]+3*rowH[1]+0.18, w:tw-0.25, h:0.44, fontSize:10.5, color:C.tealBright, valign:"middle", fontFace:"Calibri", margin:0
  });
}

// ─────────────────────────────────────────────────────────────
// SLIDE 18 — CROSS-EXPERIMENT SYNTHESIS
// ─────────────────────────────────────────────────────────────
{
  const sl = pres.addSlide();
  lightSlide(sl, "Cross-Experiment Synthesis", 18);

  const columns = [
    { color:C.navyMid, title:"What Stayed Consistent",
      items:[
        "LightGBM was the most reliable model across all four experiments",
        "Engineered behavioral features repeatedly appeared as the strongest predictors",
        "ROC-AUC remained stable even when the experimental setup changed significantly",
        "Gradient boosting models proved robust to different data conditions",
      ]},
    { color:C.green, title:"What Improved",
      items:[
        "Downsampling increased minority-class learning → recall improved for all models",
        "Reduced features preserved strong discrimination with lower complexity",
        "Threshold reduction sharply increased fraud capture (recall ↑ to 0.9+)",
        "Feature reduction led to marginal AUC improvements in LightGBM and CatBoost",
      ]},
    { color:C.amber, title:"What This Means",
      items:[
        "Fraud detection is an operational optimization problem, not just a modeling task",
        "Model choice, sampling strategy, and threshold must be tuned together holistically",
        "No single metric tells the complete deployment story — context is everything",
        "A compact, well-calibrated model can outperform a complex one at lower cost",
      ]},
  ];

  const x0 = AW+M, cw = 2.88, ch = 3.95, gx = 0.17, cy = 1.05;
  columns.forEach(({ color, title, items }, ci) => {
    const cx = x0 + ci*(cw+gx);
    card(sl, cx, cy, cw, ch);
    sl.addShape(pres.shapes.RECTANGLE, { x:cx, y:cy, w:cw, h:0.45, fill:{color}, line:{width:0} });
    sl.addText(title, { x:cx+0.12, y:cy, w:cw-0.20, h:0.45, fontSize:11.5, bold:true, color:C.white, valign:"middle", fontFace:"Calibri", margin:0 });
    items.forEach((item, ii) => {
      const iy = cy + 0.55 + ii*0.83;
      sl.addShape(pres.shapes.RECTANGLE, { x:cx+0.18, y:iy+0.08, w:0.10, h:0.10, fill:{color}, line:{width:0} });
      sl.addText(item, { x:cx+0.36, y:iy, w:cw-0.48, h:0.72, fontSize:10.5, color:C.textMid, valign:"top", fontFace:"Calibri", margin:0 });
    });
  });

  // Synthesis callout
  sl.addShape(pres.shapes.RECTANGLE, { x:AW+M, y:H-0.45, w:W-AW-2*M, h:0.38, fill:{color:C.navyDark}, line:{width:0} });
  sl.addText("Strong fraud detection depends not on the model alone, but on the full decision pipeline: features, balance, and threshold.", {
    x:AW+M+0.15, y:H-0.45, w:W-AW-2*M-0.20, h:0.38, fontSize:10.5, bold:true, color:C.tealBright, valign:"middle", fontFace:"Calibri", margin:0
  });
}

// ─────────────────────────────────────────────────────────────
// SLIDE 19 — CONCLUSIONS & FUTURE WORK (SHAP fixed)
// ─────────────────────────────────────────────────────────────
{
  const sl = pres.addSlide();
  lightSlide(sl, "Conclusions & Future Research", 19);

  const conclusions = [
    { n:1, text:"Gradient boosting models are highly effective for fraud detection on the IEEE-CIS dataset, achieving strong discrimination without deep learning complexity." },
    { n:2, text:"LightGBM was the most dependable model overall, with ROC-AUC ≈ 0.918–0.919 across configurations — making it the recommended baseline choice." },
    { n:3, text:"Behavioral feature engineering materially improved model insight and performance. Engineered features consistently ranked among top predictors via SHAP analysis." },
    { n:4, text:"SHAP explainability was implemented for all models, delivering both global feature rankings and per-prediction local attribution — bridging accuracy and interpretability." },
  ];

  const lx = AW+M, lw = 5.55, y0 = 1.05, ih = 0.92, ig = 0.14;
  card(sl, lx, y0, lw, 4.35);
  sl.addShape(pres.shapes.RECTANGLE, { x:lx, y:y0, w:lw, h:0.45, fill:{color:C.navyDark}, line:{width:0} });
  sl.addText("KEY CONCLUSIONS", { x:lx+0.18, y:y0, w:lw-0.25, h:0.45, fontSize:12, bold:true, color:C.white, valign:"middle", fontFace:"Calibri", margin:0 });
  conclusions.forEach(({ n, text }, i) => {
    const iy = y0 + 0.58 + i*(ih+ig);
    sl.addShape(pres.shapes.OVAL, { x:lx+0.18, y:iy+0.08, w:0.42, h:0.42, fill:{color:C.teal}, line:{width:0} });
    sl.addText(String(n), { x:lx+0.18, y:iy+0.08, w:0.42, h:0.42, fontSize:14, bold:true, color:C.white, align:"center", valign:"middle", fontFace:"Calibri", margin:0 });
    sl.addText(text, { x:lx+0.70, y:iy+0.04, w:lw-0.85, h:0.82, fontSize:10.8, color:C.textMid, valign:"top", fontFace:"Calibri", margin:0 });
  });

  // Right: future work (SHAP is NOT in future work — it's done)
  const rx = lx + lw + 0.28, rw = 3.30;
  card(sl, rx, y0, rw, 4.35);
  sl.addShape(pres.shapes.RECTANGLE, { x:rx, y:y0, w:rw, h:0.45, fill:{color:C.navyDark}, line:{width:0} });
  sl.addText("FUTURE RESEARCH", { x:rx+0.18, y:y0, w:rw-0.25, h:0.45, fontSize:12, bold:true, color:C.white, valign:"middle", fontFace:"Calibri", margin:0 });

  const future = [
    { color:C.purple, title:"Advanced Imbalance Methods",   desc:"Cost-sensitive learning, focal losses, SMOTE variants on harder datasets" },
    { color:C.teal,   title:"Real-Time SHAP Deployment",    desc:"Embed SHAP scoring in live inference pipelines for regulatory compliance" },
    { color:C.green,  title:"Ensemble & Hybrid Systems",    desc:"Combine rule-based and ML approaches for broader fraud coverage" },
    { color:C.amber,  title:"Model Drift Monitoring",       desc:"Longitudinal monitoring of model performance as fraud tactics evolve" },
  ];
  future.forEach(({ color, title, desc }, i) => {
    const fy = y0 + 0.60 + i*0.90;
    accentCard(sl, rx+0.15, fy, rw-0.20, 0.78, color, false);
    sl.addText(title, { x:rx+0.28, y:fy+0.06, w:rw-0.42, h:0.30, fontSize:11.5, bold:true, color:C.textDark, valign:"middle", fontFace:"Calibri", margin:0 });
    sl.addText(desc,  { x:rx+0.28, y:fy+0.38, w:rw-0.42, h:0.32, fontSize:10,   color:C.textMid,  valign:"top",    fontFace:"Calibri", margin:0 });
  });
}

// ─────────────────────────────────────────────────────────────
// SLIDE 20 — THANK YOU
// ─────────────────────────────────────────────────────────────
{
  const sl = pres.addSlide();
  darkBg(sl);

  // Left accent
  sl.addShape(pres.shapes.RECTANGLE, { x:0, y:0, w:0.12, h:H, fill:{color:C.teal}, line:{width:0} });

  sl.addText("THANK YOU", {
    x:0.55, y:0.80, w:7.0, h:1.6,
    fontSize:64, bold:true, color:C.white, align:"left", valign:"middle",
    fontFace:"Calibri", margin:0
  });
  sl.addText("Questions?", {
    x:0.55, y:2.35, w:5.0, h:0.65,
    fontSize:28, color:C.tealBright, align:"left", valign:"middle",
    fontFace:"Calibri", margin:0
  });

  // Thin line
  sl.addShape(pres.shapes.LINE, { x:0.55, y:3.15, w:6.5, h:0, line:{color:C.teal, width:1.5} });

  // University info
  sl.addText([
    { text:"Konstantinos Koutsompinas", options:{bold:true, breakLine:true} },
    { text:"Supervisor: Athanasios Argyriou", options:{breakLine:true} },
    { text:"National & Kapodistrian University of Athens" },
  ], {
    x:0.55, y:3.30, w:7.0, h:1.05,
    fontSize:11, color:"94A3B8", fontFace:"Calibri", align:"left", valign:"top", margin:0
  });

  // Right side: final stat panel
  const rx = 7.7, rw = 1.95;
  sl.addShape(pres.shapes.RECTANGLE, { x:rx, y:0.80, w:rw, h:3.8, fill:{color:C.navyMid}, line:{color:C.teal, width:1.5}, shadow:shH() });

  sl.addText("Best ROC-AUC", { x:rx, y:0.96, w:rw, h:0.32, fontSize:9.5, color:C.tealBright, align:"center", fontFace:"Calibri", margin:0 });
  sl.addText("0.919", { x:rx, y:1.25, w:rw, h:0.75, fontSize:36, bold:true, color:C.white, align:"center", valign:"middle", fontFace:"Calibri", margin:0 });
  sl.addText("LightGBM\nReduced Features", { x:rx, y:1.98, w:rw, h:0.50, fontSize:9.5, color:"94A3B8", align:"center", fontFace:"Calibri", margin:0 });

  sl.addShape(pres.shapes.LINE, { x:rx+0.25, y:2.58, w:rw-0.50, h:0, line:{color:C.navyDark, width:1} });

  sl.addText("Best Recall\n(Thr. 0.1)", { x:rx, y:2.70, w:rw, h:0.45, fontSize:9.5, color:C.tealBright, align:"center", fontFace:"Calibri", margin:0 });
  sl.addText("0.922", { x:rx, y:3.12, w:rw, h:0.65, fontSize:30, bold:true, color:C.white, align:"center", valign:"middle", fontFace:"Calibri", margin:0 });
  sl.addText("XGBoost", { x:rx, y:3.75, w:rw, h:0.30, fontSize:9.5, color:"94A3B8", align:"center", fontFace:"Calibri", margin:0 });

  // MSc tag
  sl.addText("MSc in Business Administration, Analytics & IS  |  February 2026", {
    x:0.55, y:H-0.38, w:W-0.70, h:0.30,
    fontSize:8.5, color:"475569", align:"center", fontFace:"Calibri", margin:0
  });
}

// ─────────────────────────────────────────────────────────────
// OUTPUT
// ─────────────────────────────────────────────────────────────
pres.writeFile({ fileName: "/sessions/jolly-kind-einstein/Behavioral_Fraud_Analytics_V2.pptx" })
  .then(() => console.log("✅  V2 Presentation saved successfully — 20 slides"))
  .catch(err => console.error("❌  Error:", err));