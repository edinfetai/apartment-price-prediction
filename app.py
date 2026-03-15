import gradio as gr
import joblib
import pandas as pd
from math import radians, sin, cos, sqrt, atan2

# ----------------------------
# Load trained model and dataset
# ----------------------------
model = joblib.load("best_model.pkl")
df = pd.read_csv("apartments_data_enriched_with_new_features.csv")

# Basic cleanup
df["town"] = df["town"].astype(str).str.strip()

# Zurich center coordinates
ZURICH_LAT = 47.3769
ZURICH_LON = 8.5417

# ----------------------------
# Build postal/town mappings from dataset
# ----------------------------
postal_to_town_df = (
    df.groupby(["postalcode", "town"])
    .size()
    .reset_index(name="count")
    .sort_values(["postalcode", "count"], ascending=[True, False])
    .drop_duplicates("postalcode")
)

town_to_postal_df = (
    df.groupby(["town", "postalcode"])
    .size()
    .reset_index(name="count")
    .sort_values(["town", "count"], ascending=[True, False])
    .drop_duplicates("town")
)

postal_to_town = {
    int(row["postalcode"]): str(row["town"])
    for _, row in postal_to_town_df.iterrows()
}
town_to_postal = {
    str(row["town"]): int(row["postalcode"])
    for _, row in town_to_postal_df.iterrows()
}

postal_choices = sorted(postal_to_town.keys())
town_choices = sorted(town_to_postal.keys())

# Representative coordinates per postal code
postal_geo_df = (
    df.groupby("postalcode")[["lat", "lon"]]
    .median()
    .reset_index()
)

postal_to_geo = {
    int(row["postalcode"]): (float(row["lat"]), float(row["lon"]))
    for _, row in postal_geo_df.iterrows()
}

# ----------------------------
# Helpers
# ----------------------------
def haversine_distance(lat1, lon1, lat2, lon2):
    r = 6371  # km
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return r * c


def update_town_from_postal(postalcode):
    if postalcode is None:
        return gr.update()
    return gr.update(value=postal_to_town.get(int(postalcode), ""))


def update_postal_from_town(town):
    if not town:
        return gr.update()
    return gr.update(value=town_to_postal.get(str(town), None))


def format_feature_summary(features_selected):
    if not features_selected:
        return "No extra features selected"
    return "Selected: " + ", ".join(features_selected)


def predict_price(rooms, area, postalcode, town, features_selected):
    try:
        if rooms is None or area is None:
            return (
                "Please enter values for rooms and area.",
                "—",
                "Please complete all required fields."
            )

        rooms = float(rooms)
        area = float(area)

        if area <= 0:
            return (
                "Area must be greater than 0.",
                "—",
                "Please enter a valid apartment size."
            )

        if postalcode is None:
            return (
                "Please select a postal code.",
                "—",
                "Postal code is required."
            )

        postalcode = int(postalcode)

        if not town:
            town = postal_to_town.get(postalcode, "")

        # coordinates in background
        lat, lon = postal_to_geo.get(postalcode, (47.3769, 8.5417))

        # feature chips -> booleans
        features_selected = features_selected or []
        furnished = "Furnished" in features_selected
        temporary = "Temporary" in features_selected
        luxurious = "Luxurious" in features_selected
        zurich_city = "Zurich City" in features_selected

        # use first row as feature template
        row = df.iloc[0].copy()

        # overwrite user-controlled values
        row["rooms"] = rooms
        row["area"] = area
        row["postalcode"] = postalcode
        row["town"] = town
        row["lat"] = lat
        row["lon"] = lon
        row["furnished"] = furnished
        row["temporary"] = temporary
        row["luxurious"] = luxurious
        row["zurich_city"] = zurich_city

        # recompute engineered values
        row["room_per_m2"] = rooms / area if area > 0 else 0
        row["distance_to_center"] = haversine_distance(lat, lon, ZURICH_LAT, ZURICH_LON)

        # unknown at prediction time
        if "price_per_m2" in row.index:
            row["price_per_m2"] = 0

        input_data = pd.DataFrame([row])

        if "price" in input_data.columns:
            input_data = input_data.drop(columns=["price"])

        prediction = float(model.predict(input_data)[0])

        price_text = f"CHF {prediction:,.2f}"
        meta_text = f"{rooms:g} rooms • {area:g} m² • {postalcode} {town}"
        detail_text = format_feature_summary(features_selected)

        return price_text, meta_text, detail_text

    except Exception as e:
        return (
            "Prediction failed",
            "—",
            f"Error: {str(e)}"
        )


# ----------------------------
# Futuristic styling
# ----------------------------
custom_css = """
:root {
  --bg1: #060816;
  --bg2: #140b2d;
  --bg3: #0a1228;
  --glass: rgba(15, 23, 42, 0.60);
  --glass-strong: rgba(15, 23, 42, 0.82);
  --border: rgba(168, 85, 247, 0.22);
  --text: #f8fafc;
  --muted: #cbd5e1;
  --cyan: #22d3ee;
  --purple: #a855f7;
  --pink: #ec4899;
}

body, .gradio-container {
  background:
    radial-gradient(circle at 20% 20%, rgba(168,85,247,0.22), transparent 30%),
    radial-gradient(circle at 80% 25%, rgba(34,211,238,0.18), transparent 28%),
    radial-gradient(circle at 70% 80%, rgba(236,72,153,0.16), transparent 24%),
    linear-gradient(135deg, var(--bg1) 0%, var(--bg2) 45%, var(--bg3) 100%) !important;
  color: var(--text) !important;
  font-family: Inter, Arial, sans-serif !important;
}

.gradio-container {
  max-width: 1280px !important;
  margin: 0 auto !important;
  padding-top: 22px !important;
  padding-bottom: 28px !important;
}

#hero-card,
#form-card,
#result-card,
#feature-card {
  background: var(--glass) !important;
  backdrop-filter: blur(16px) !important;
  -webkit-backdrop-filter: blur(16px) !important;
  border: 1px solid var(--border) !important;
  border-radius: 28px !important;
  box-shadow:
    0 10px 30px rgba(0,0,0,0.28),
    inset 0 1px 0 rgba(255,255,255,0.04) !important;
}

#hero-card {
  padding: 24px 28px 10px 28px !important;
  margin-bottom: 18px !important;
}

#form-card, #result-card {
  padding: 22px !important;
}

#feature-card {
  padding: 18px !important;
}

h1, h2, h3, p, label, span, div {
  color: var(--text) !important;
}

.hero-title {
  font-size: 40px !important;
  font-weight: 800 !important;
  line-height: 1.1 !important;
  margin: 0 !important;
  background: linear-gradient(90deg, #67e8f9, #c084fc, #f9a8d4);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}

.hero-subtitle {
  margin-top: 10px !important;
  color: var(--muted) !important;
  font-size: 16px !important;
}

.section-title {
  font-size: 20px !important;
  font-weight: 700 !important;
  margin-bottom: 8px !important;
}

.label-soft {
  color: #c4b5fd !important;
  font-size: 13px !important;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin-bottom: 8px !important;
}

.gr-textbox, .gr-number, .gr-dropdown {
  border-radius: 18px !important;
}

input, textarea, select {
  background: rgba(15, 23, 42, 0.72) !important;
  border: 1px solid rgba(168,85,247,0.28) !important;
  color: white !important;
  border-radius: 18px !important;
  min-height: 54px !important;
  font-size: 17px !important;
}

input:focus, textarea:focus, select:focus {
  border-color: rgba(34,211,238,0.75) !important;
  box-shadow: 0 0 0 3px rgba(34,211,238,0.10) !important;
}

button.primary, button.lg.primary {
  background: linear-gradient(90deg, #06b6d4, #8b5cf6, #ec4899) !important;
  border: none !important;
  color: white !important;
  font-weight: 800 !important;
  border-radius: 18px !important;
  min-height: 56px !important;
  font-size: 16px !important;
  box-shadow: 0 10px 24px rgba(139,92,246,0.28) !important;
}

button.primary:hover, button.lg.primary:hover {
  filter: brightness(1.08);
  transform: translateY(-1px);
}

.gr-checkboxgroup {
  gap: 10px !important;
}

.gr-checkboxgroup label {
  background: rgba(15, 23, 42, 0.72) !important;
  border: 1px solid rgba(168,85,247,0.22) !important;
  border-radius: 999px !important;
  padding: 10px 14px !important;
  margin-right: 8px !important;
  margin-bottom: 10px !important;
  display: inline-flex !important;
  align-items: center !important;
  transition: all .18s ease !important;
}

.gr-checkboxgroup label:hover {
  border-color: rgba(34,211,238,0.55) !important;
  box-shadow: 0 0 18px rgba(34,211,238,0.10) !important;
}

.result-price textarea,
.result-price input {
  font-size: 42px !important;
  font-weight: 800 !important;
  text-align: center !important;
  color: white !important;
  min-height: 88px !important;
  background:
    linear-gradient(90deg, rgba(34,211,238,0.10), rgba(168,85,247,0.10), rgba(236,72,153,0.10)) !important;
}

.result-meta textarea,
.result-detail textarea {
  text-align: center !important;
  color: #dbeafe !important;
}

footer {
  display: none !important;
}
"""


# ----------------------------
# UI
# ----------------------------
with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    with gr.Group(elem_id="hero-card"):
        gr.Markdown(
            """
            <div class="hero-title">Apartment Price Prediction Zurich</div>
            <div class="hero-subtitle">
              Smart, modern rental estimation for the canton of Zurich.
              Enter a few apartment details and get an instant prediction.
            </div>
            """
        )

    with gr.Row():
        with gr.Column(scale=7):
            with gr.Group(elem_id="form-card"):
                gr.Markdown('<div class="section-title">Apartment Details</div>')

                rooms = gr.Number(label="Rooms", value=3, precision=1)
                area = gr.Number(label="Area (m²)", value=70, precision=1)

                postalcode = gr.Dropdown(
                    choices=postal_choices,
                    value=8001 if 8001 in postal_choices else postal_choices[0],
                    label="Postal Code",
                    interactive=True
                )

                town = gr.Dropdown(
                    choices=town_choices,
                    value=postal_to_town.get(8001, town_choices[0]),
                    label="Town",
                    interactive=True
                )

            with gr.Group(elem_id="feature-card"):
                gr.Markdown('<div class="section-title">Apartment Features</div>')

                features_selected = gr.CheckboxGroup(
                    choices=["Furnished", "Temporary", "Luxurious", "Zurich City"],
                    value=["Zurich City"],
                    label=""
                )

                predict_button = gr.Button("Predict Rent", variant="primary")

        with gr.Column(scale=5):
            with gr.Group(elem_id="result-card"):
                gr.Markdown('<div class="section-title">Monthly Rent Prediction</div>')

                output_price = gr.Textbox(
                    label="Estimated Rent",
                    value="CHF —",
                    interactive=False,
                    elem_classes=["result-price"]
                )

                output_meta = gr.Textbox(
                    label="Summary",
                    value="Select your apartment details",
                    interactive=False,
                    elem_classes=["result-meta"]
                )

                output_detail = gr.Textbox(
                    label="Features",
                    value="Selected: Zurich City",
                    interactive=False,
                    elem_classes=["result-detail"]
                )

    # sync postal -> town
    postalcode.change(
        fn=update_town_from_postal,
        inputs=postalcode,
        outputs=town
    )

    # sync town -> postal
    town.change(
        fn=update_postal_from_town,
        inputs=town,
        outputs=postalcode
    )

    predict_button.click(
        fn=predict_price,
        inputs=[rooms, area, postalcode, town, features_selected],
        outputs=[output_price, output_meta, output_detail]
    )

demo.launch()