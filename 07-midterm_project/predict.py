import pickle
import uvicorn
from fastapi import FastAPI, Request, Query
from fastapi.responses import JSONResponse

MODEL_FILE = "pipeline.bin"


with open(MODEL_FILE, "rb") as f_in:
    dv, model = pickle.load(f_in)

app = FastAPI(title="Adult Income Prediction API")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(
    request: Request,
    threshold: float = Query(0.5, ge=0.0, le=1.0, description="Threshold for classifying >50K income"),
):
    person = await request.json()
    X = dv.transform([person])
    prob_gt_50k = float(model.predict_proba(X)[0, 1])
    label = ">50K" if prob_gt_50k >= threshold else "<=50K"

    result = {"probability_gt_50k": prob_gt_50k, "threshold": threshold, "predicted_label": label}
    return JSONResponse(result)


if __name__ == "__main__":
    uvicorn.run("predict:app", host="0.0.0.0", port=9696, reload=True)
