from flask import Flask, jsonify, request
import os, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from xgboost import XGBClassifier

# ----------------- إعدادات عامة -----------------
SEED = 42
np.random.seed(SEED)
ART_DIR = Path("artifacts"); ART_DIR.mkdir(exist_ok=True)
MODEL_PATH = ART_DIR/"model.pkl"
SCALER_PATH = ART_DIR/"scaler.pkl"
REPORT_PATH = ART_DIR/"report.json"
DATA_PATH = Path("data.csv")  

# تقسيم الخطر (20% لكل فئة)
BINS = [
    (0.00, 0.20, "سليم جدًا",   "استمر على نمط حياتك الصحي."),
    (0.20, 0.40, "سليم",        "حافظ على التمارين والغذاء الجيد، وراقب صحتك."),
    (0.40, 0.60, "معرّض للخطر", "يُفضل إجراء فحوصات دورية للتأكد من سلامتك."),
    (0.60, 0.80, "مريض",        "استشر الطبيب قريبًا وأجرِ الفحوصات اللازمة."),
    (0.80, 1.00, "خطر شديد",    "راجع الطبيب فورًا ولا تؤجل.")
]

def bin_info(p: float):
    p = max(0.0, min(1.0, float(p)))
    for lo, hi, name, advice in BINS:
        if p < hi or abs(hi-1.0) < 1e-9:
            return {
                "category": name,
                "advice": advice,
                "range": {"low": round(lo,2), "high": round(hi,2)}
            }
    return {"category":"غير محدد","advice":"—","range":{"low":0.0,"high":1.0}}

# ----------------- توليد بيانات محاكاة -----------------
def generate_synthetic(n=5000, pos_rate=0.30):
    rng = np.random.default_rng(SEED)
    # 10 حساسات
    s = [rng.normal(loc, scale, n) for loc, scale in [
        (0.0,1.0),(0.5,1.1),(-0.3,1.2),(0,2),(0.2,0.8),
        (0,1),(0.1,0.9),(0.4,1.2),(-0.2,1.5),(0.3,1.3)
    ]]
    X = np.vstack(s).T

    # دالة خطر غير خطية
    z = 0.5*s[0] + 0.6*s[1] - 0.4*s[2] + 0.3*s[3]*s[1] - 0.5*s[4]
    z += 0.2*np.tanh(s[5]) - 0.3*(s[6]*2) + 0.4*(s[7]>0.5) + 0.3*s[8] - 0.6*s[9]
    z = (z - z.mean())/(z.std()+1e-8) + rng.normal(0, 0.6, n)
    p = 1/(1+np.exp(-z))

    # ضبط الانتشار ليقارب pos_rate
    def shift_to_prevalence(target, probs):
        lo, hi = -5, 5
        for _ in range(40):
            m = (lo+hi)/2
            logits = np.log(probs/(1-probs)) + m
            mean_pos = (1/(1+np.exp(-logits))).mean()
            if mean_pos < target: hi = m
            else: lo = m
        return (lo+hi)/2
    b = shift_to_prevalence(pos_rate, p)
    logits = np.log(p/(1-p)) + b
    p_adj = 1/(1+np.exp(-logits))
    y = (rng.random(n) < p_adj).astype(int)

    cols = [f"sensor{i+1}" for i in range(10)]
    df = pd.DataFrame(X, columns=cols); df["danger"] = y
    return df

# ----------------- تدريب وتقرير -----------------
def train_and_report(df: pd.DataFrame):
    feature_cols = [f"sensor{i+1}" for i in range(10)]
    X = df[feature_cols].values
    y = df["danger"].values

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)

    scaler = StandardScaler()
    X_trs = scaler.fit_transform(X_tr)
    X_tes = scaler.transform(X_te)

    pos = y_tr.sum()
    neg = len(y_tr) - pos
    spw = float(neg / (pos + 1e-8))

    base = XGBClassifier(
        random_state=SEED, tree_method="hist", eval_metric="logloss",
        use_label_encoder=False, n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0, scale_pos_weight=spw
    )

    param_dist = {
        "max_depth": [4,5,6],
        "learning_rate": [0.03,0.05,0.08],
        "n_estimators": [200,300,400],
        "subsample": [0.8,0.9,1.0],
        "colsample_bytree": [0.7,0.85,1.0],
        "reg_lambda": [0.5,1.0,1.5]
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    search = RandomizedSearchCV(base, param_distributions=param_dist, n_iter=10,
                                scoring="roc_auc", cv=cv, n_jobs=-1, random_state=SEED, verbose=0)
    search.fit(X_trs, y_tr)
    model = search.best_estimator_

    proba = model.predict_proba(X_tes)[:,1]
    pred  = (proba >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
    specificity = tn / (tn + fp) if (tn+fp)>0 else 0.0

    report = {
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": "XGBoost",
        "best_params": search.best_params_,
        "metrics": {
            "accuracy": float(accuracy_score(y_te, pred)),
            "precision": float(precision_score(y_te, pred, zero_division=0)),
            "recall_sensitivity": float(recall_score(y_te, pred, zero_division=0)),
            "specificity": float(specificity),
            "f1": float(f1_score(y_te, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_te, proba)),
            "pr_auc": float(average_precision_score(y_te, proba)),
            "confusion_matrix": {"tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp)},
            "test_size": int(len(y_te))
        }
    }

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return model, scaler, report

# ----------------- تجهيز أولي -----------------
def ensure_ready():
    if MODEL_PATH.exists() and SCALER_PATH.exists():
        return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)
    df = generate_synthetic()
    model, scaler, _ = train_and_report(df)
    return model, scaler

model, scaler = ensure_ready()

# ----------------- Flask -----------------
app = Flask("_name_")

@app.get("/")
def root():
    return jsonify({
        "service":"medical-risk-api",
        "note":"النتائج للتوجيه فقط — القرار النهائي من الطبيب.",
        "endpoints":["/predict (POST)","/upload-batch (POST)","/retrain (POST)","/metrics (GET)","/thresholds (GET)"]
    })

@app.get("/thresholds")
def thresholds():
    return jsonify(BINS)

@app.post("/predict")
def predict():
    data = request.get_json(force=True, silent=True) or {}
    sensors = data.get("sensors")
    if not isinstance(sensors, (list,tuple)) or len(sensors)!=10:
        return jsonify({"error":"أرسل JSON فيه 'sensors' = قائمة 10 أرقام"}), 400
    try:
        X = np.asarray(sensors, dtype=float).reshape(1, -1)
    except Exception as e:
        return jsonify({"error":"قيم غير صالحة","details":str(e)}), 400

    global model, scaler
    if not isinstance(model, XGBClassifier) or scaler is None:
        model = joblib.load(MODEL_PATH); scaler = joblib.load(SCALER_PATH)

    Xs = scaler.transform(X)
    p = float(model.predict_proba(Xs)[:,1][0])
    info = bin_info(p)
    return jsonify({
        "probability": round(p,4),
        "prediction": info["category"],
        "advice": info["advice"],
        "bin_range": info["range"],
        "note":"هذا التصنيف للتوجيه فقط — القرارات الطبية النهائية للطبيب."
    })

@app.post("/upload-batch")
def upload_batch():
    data = request.get_json(force=True, silent=True) or {}
    rows = data.get("rows"); has_label = bool(data.get("has_label", True))
    if not isinstance(rows, (list,tuple)) or len(rows)==0:
        return jsonify({"error":"أرسل 'rows' كمصفوفة صفوف"}), 400
    try:
        arr = np.asarray(rows, dtype=float)
    except Exception as e:
        return jsonify({"error":"صفوف غير صالحة","details":str(e)}), 400

    if has_label:
        if arr.ndim!=2 or arr.shape[1]!=11:
            return jsonify({"error":"مع has_label=true كل صف لازم 11 قيم (10 حساسات + label)"}), 400
        cols = [f"sensor{i+1}" for i in range(10)] + ["danger"]
    else:
        if arr.ndim!=2 or arr.shape[1]!=10:
            return jsonify({"error":"مع has_label=false كل صف لازم 10 قيم"}), 400
        cols = [f"sensor{i+1}" for i in range(10)]

    df_new = pd.DataFrame(arr, columns=cols)
    if DATA_PATH.exists():
        df_old = pd.read_csv(DATA_PATH)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new
    df_all.to_csv(DATA_PATH, index=False)
    return jsonify({"saved_rows": int(len(df_new)), "total_rows": int(len(df_all)), "has_label": has_label})

@app.post("/retrain")
def retrain():
    if not DATA_PATH.exists():
        return jsonify({"error":"لا يوجد data.csv — ارفع بياناتك أولًا بـ /upload-batch"}), 400
    df = pd.read_csv(DATA_PATH)
    expected = [f"sensor{i+1}" for i in range(10)] + ["danger"]
    if not all(c in df.columns for c in expected):
        return jsonify({"error":"data.csv لازم يحتوي 10 حساسات + عمود danger (0/1)"}), 400

    m, s, rep = train_and_report(df[expected])
    global model, scaler
    model, scaler = m, s
    return jsonify({"status":"retrained", "report": rep})

@app.get("/metrics")
def metrics():
    if REPORT_PATH.exists():
        with open(REPORT_PATH, "r", encoding="utf-8") as f:
            return jsonify(json.load(f))
    return jsonify({"error":"لا يوجد تقرير — درّب الموديل أولًا"}), 404

# ----------------- Run -----------------
if "_name_" == "_main_":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
