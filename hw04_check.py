import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score

URL = 'https://raw.githubusercontent.com/alexeygrigorev/datasets/master/course_lead_scoring.csv'
df = pd.read_csv(URL)
TARGET = 'converted'
features = [c for c in df.columns if c != TARGET]
num_cols = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
cat_cols = [c for c in features if c not in num_cols]
# Impute
for c in cat_cols:
    df[c] = df[c].astype('object').fillna('NA')
for c in num_cols:
    df[c] = df[c].astype('float64').fillna(0.0)

def run(split_stratify=False):
    X = df[features].copy(); y = df[TARGET].astype('int64').values
    if split_stratify:
        X_tr, X_tmp, y_tr, y_tmp = train_test_split(X,y,test_size=0.4,random_state=1,stratify=y)
        X_va, X_te, y_va, y_te = train_test_split(X_tmp,y_tmp,test_size=0.5,random_state=1,stratify=y_tmp)
    else:
        X_tr, X_tmp, y_tr, y_tmp = train_test_split(X,y,test_size=0.4,random_state=1)
        X_va, X_te, y_va, y_te = train_test_split(X_tmp,y_tmp,test_size=0.5,random_state=1)
    # Q1 AUCs numeric
    def auc_num(s,y):
        from sklearn.metrics import roc_auc_score
        a = roc_auc_score(y, s)
        if a < 0.5:
            a = roc_auc_score(y, -s)
        return float(a)
    aucs = {c: auc_num(X_tr[c].values, y_tr) for c in num_cols}
    asked = ['lead_score','number_of_courses_viewed','interaction_count','annual_income']
    aucs_f = {k: aucs.get(k, np.nan) for k in asked}
    # DictVectorizer
    dv = DictVectorizer(sparse=False)
    X_tr_dv = dv.fit_transform(X_tr.to_dict('records'))
    X_va_dv = dv.transform(X_va.to_dict('records'))
    m = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=1)
    m.fit(X_tr_dv, y_tr)
    val_scores = m.predict_proba(X_va_dv)[:,1]
    val_auc = roc_auc_score(y_va, val_scores)
    # thresholds
    thr = np.linspace(0,1,101)
    P,R=[],[]
    for t in thr:
        yb = (val_scores>=t).astype(int)
        P.append(precision_score(y_va,yb,zero_division=0))
        R.append(recall_score(y_va,yb))
    idx = int(np.argmin(np.abs(np.array(P)-np.array(R))))
    t_int = float(thr[idx])
    # F1
    f1s = [0 if p+r==0 else 2*p*r/(p+r) for p,r in zip(P,R)]
    idxf = int(np.argmax(f1s))
    t_f1 = float(thr[idxf]); f1max = float(f1s[idxf])
    # 5-fold CV
    df_full = pd.concat([X_tr.assign(converted=y_tr), X_va.assign(converted=y_va)]).reset_index(drop=True)
    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    aucs_cv=[]
    for tr_idx, va_idx in kf.split(df_full):
        dtr = df_full.iloc[tr_idx]; dva = df_full.iloc[va_idx]
        dvv=DictVectorizer(sparse=False)
        Xtr=dvv.fit_transform(dtr[features].to_dict('records'))
        Xva=dvv.transform(dva[features].to_dict('records'))
        m=LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=1)
        m.fit(Xtr,dtr['converted'].values)
        pr=m.predict_proba(Xva)[:,1]
        aucs_cv.append(roc_auc_score(dva['converted'].values, pr))
    std_cv=float(np.std(aucs_cv))
    # tuning
    res={}
    for C in [1e-6,1e-3,1]:
        scores=[]
        for tr_idx, va_idx in kf.split(df_full):
            dtr=df_full.iloc[tr_idx]; dva=df_full.iloc[va_idx]
            dvv=DictVectorizer(sparse=False)
            Xtr=dvv.fit_transform(dtr[features].to_dict('records'))
            Xva=dvv.transform(dva[features].to_dict('records'))
            m=LogisticRegression(solver='liblinear', C=C, max_iter=1000, random_state=1)
            m.fit(Xtr,dtr['converted'].values)
            pr=m.predict_proba(Xva)[:,1]
            scores.append(roc_auc_score(dva['converted'].values, pr))
        res[C]={'mean':round(float(np.mean(scores)),3),'std':round(float(np.std(scores)),3)}
    best_mean=max(v['mean'] for v in res.values())
    cands=[C for C,v in res.items() if v['mean']==best_mean]
    if len(cands)>1:
        best_std=min(res[C]['std'] for C in cands)
        cands=[C for C in cands if res[C]['std']==best_std]
    bestC=min(cands)
    return {
        'split_stratified': split_stratify,
        'Q1': aucs_f,
        'Q2_val_auc': round(val_auc,3),
        'Q3_t_intersection': t_int,
        'Q4_t_best_f1': t_f1,
        'Q5_std_cv': round(std_cv,3),
        'Q6': res, 'Q6_bestC': bestC
    }

print('Running without stratify...')
res1=run(split_stratify=False)
print(res1)
print('Running with stratify...')
res2=run(split_stratify=True)
print(res2)
