# Imports
import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
from matplotlib.patches import Polygon
import io

# Models cache - makes model retrieval quicker when used once
MODELS_CACHE = {}

# Use full page
st.set_page_config(layout="wide")

# Make page scrollable
st.markdown(
    """
    <style>
    html, body, [class*="stAppViewContainer"], [class*="main"], [class*="block-container"] {
        height: 100%;
        overflow: auto !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# This function takes user-input stuff row and expands it into an 80x80 grid (6,400 points) on a location plot - stuff features are constant across all location points
# Everything downstream operates on that grid, not the original single row.
def generate_grid(row, pitch, steps=80):

    # Tighten bounds to try and include only competitive pitch locations
    platex_range = np.linspace(-2, 2, steps)
    platez_range = np.linspace(-0.5, 4, steps)
    platex_grid, platez_grid = np.meshgrid(platex_range, platez_range)

    # Don't include fb_velo as feature in fastball-adjacent pitches
    if pitch in ["FB", "FC"]:
        df_grid = pd.DataFrame({
            'release_speed':       row['release_speed'].iloc[0],
            'release_height':      row['release_height'].iloc[0],
            'release_side':        row['release_side'].iloc[0],
            'arm_angle':           row['arm_angle'].iloc[0],
            'release_extension':   row['release_extension'].iloc[0],
            'ivb':                 row['ivb'].iloc[0],
            'hb':                  row['hb'].iloc[0],
            'spin_rate':           row['spin_rate'].iloc[0],
            'plate_x':             platex_grid.ravel(),
            'plate_z':             platez_grid.ravel(),
        })

    # Include fb_velo as feature in non-fastball pitches
    else:
        df_grid = pd.DataFrame({
            'release_speed':       row['release_speed'].iloc[0],
            'release_height':      row['release_height'].iloc[0],
            'release_side':        row['release_side'].iloc[0],
            'arm_angle':           row['arm_angle'].iloc[0],
            'release_extension':   row['release_extension'].iloc[0],
            'ivb':                 row['ivb'].iloc[0],
            'hb':                  row['hb'].iloc[0],
            'spin_rate':           row['spin_rate'].iloc[0],
            'plate_x':             platex_grid.ravel(),
            'plate_z':             platez_grid.ravel(),
            'fb_velo':             row['fb_velo'].iloc[0],
        })

    # Return our plate_x range, plate_z range, our newly generated grid, and its shape
    return platex_range, platez_range, df_grid, platez_grid.shape

# This function calls in a pre-trained model and predicts the run value of each stuff + location combination on our grid
# Run values are then converted to a more interpretable Pitch+ value
def predict_loc_grid(df_grid, grid_shape, pitch, p_hand, b_side, features, pitches_plus):

    # If our model has not already been used (cached)
    if (pitch, p_hand, b_side) not in MODELS_CACHE:
        # Open pre-trained model for equivalent pitch group / pitcher handedness / batter stance
        with open(f'models/train_and_test/{pitch}/{pitch}_{p_hand}HP_{b_side}HB_alldata.pkl', 'rb') as f:
            # Load model
            MODELS_CACHE[(pitch, p_hand, b_side)] = pickle.load(f)
    # If our model has already been used, call it back
    model = MODELS_CACHE[(pitch, p_hand, b_side)]

    # Predict run value for each pitch in grid
    preds = model.predict(df_grid[features])

    # From our pre-caculated dataset of expected run values of pitcher / pitch type / batter stance combinations from 2026 season
    # Calculate population mean and standard deviation
    pop_mean = pitches_plus["xRV"].mean()
    pop_std = pitches_plus["xRV"].std()

    # Convert our grid's expected run values to Pitch+, where 100 is a league average pitch, and every +/- 10 is a standard deviation
    pitches_plus_grid = 100 + (-10 * (preds - pop_mean) / pop_std)
    pitches_plus_grid = pitches_plus_grid.reshape(grid_shape)

    # Apply a gaussian filter to smooth results out - helps to reduce the affects of any noise the model may have picked up on
    pitches_plus_grid = gaussian_filter(pitches_plus_grid, sigma=3)

    # Return our grid with Pitch+ values, based on expected run-values our model produced
    return pitches_plus_grid

# This function adds the batter silhouette to the plots
def add_batter(ax, b_side, x_pos=2.4, y_pos=2.9, zoom=0.7):
    img = mpimg.imread('batter_silhouette.webp')
    # Flip side and mirror image if batter is left-handed
    if b_side == "L":
        img = np.fliplr(img)   
        x_pos = -x_pos
    imagebox = OffsetImage(img, zoom=zoom)
    ab = AnnotationBbox(imagebox, (x_pos, y_pos), frameon=False, zorder=4)
    ax.add_artist(ab)

# This function adds the home plate visualization to the plots
def add_home_plate(ax, y_base=0, width=0.83, thickness=0.15):
    x0, x1 = -width, width
    pts = [
        (x0, y_base),
        (x1, y_base),
        (x1, y_base + thickness*0.3),
        (0, y_base + thickness),
        (x0, y_base + thickness*0.3),
    ]
    plate = Polygon(pts, closed=True, facecolor='white', edgecolor='black', linewidth=1.5, zorder=5)
    ax.add_patch(plate)

# This function creates our visualization - creates and populates plots, adds filters, adds extra visualizations, controls its size, etc
def plot_loc_landscape(platex_range, platez_range, preds_grid, pitch, pitch_type, p_hand, b_side, window=10):
    platex_mesh, platez_mesh = np.meshgrid(platex_range, platez_range)

    # Define the boundaries of the strike zone (average MLB strike zone)
    zone_left, zone_right = -0.83, 0.83
    zone_bottom, zone_top = 1.5, 3.5

    # Define the inside of the strike zone
    in_zone = (
        (platex_mesh >= zone_left) & (platex_mesh <= zone_right) &
        (platez_mesh >= zone_bottom) & (platez_mesh <= zone_top)
    )

    # Only show the Pitch+ metrics where Pitch+ >= 70 OR the location is within the strike zone
    show = in_zone | (preds_grid >= 70)
    masked_grid = np.where(show, preds_grid, np.nan)

    plt.close('all')
    # Initialize our plot
    fig, ax = plt.subplots(figsize=(8, 7))

    # Use heat map to map Pitch+ values on location plot
    heatmap = ax.contourf(platex_range, platez_range, masked_grid,
                      levels=np.linspace(50, 150, 51),
                      cmap='RdYlGn', vmin=50, vmax=150,
                      extend='both')
    cb = plt.colorbar(heatmap, ax=ax, label='Pitch+')
    cb.set_ticks([50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150])

    # Add strike zone visualization to plot, to further help interpret results
    strike_zone = plt.Rectangle(
        (zone_left, zone_bottom),
        zone_right - zone_left,
        zone_top - zone_bottom,
        fill=False, edgecolor='black', linewidth=2, linestyle='--'
    )
    ax.add_patch(strike_zone)

    # Adds batter and home plate visualization
    add_batter(ax, b_side)
    add_home_plate(ax)

    # Show a wider range than we calculate for, to get a more realistic view from pitcher's perspective in plot
    ax.set_xlim(-3, 3)
    ax.set_ylim(-1, 6)
    # Light gray background
    ax.set_facecolor('lightgray')
    # Labels / title
    ax.set_xlabel('Plate Loc (ft)')
    ax.set_ylabel('Plate Loc (ft)')
    ax.set_title(f'Pitch Location Landscape — {p_hand}HP {pitch_type} vs {b_side}HB')

    # Return our figure
    return fig

# This function converts our figure to a buff - allows 2 plots to be displayed side-by-side at consistent sizes
def fig_to_buf(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches=None, dpi=150)
    buf.seek(0)
    return buf

# Page title
st.title("Pitching+ Landscape on Pitch Location Plot")

# Create columns for our dropdowns/input boxes where user can fill in metrics
c1, c2, c3, c4, c5, c6, c7 = st.columns([1,1,1,1,1,1,1])
d1, d2, d3, d4, d5, d6, d7 = st.columns([1,1,1,1,1,1,1])

# Pitch type selection
with c1:
    pitches = ["FF", "SI", "FC", "CH", "SL", "CU", "ST", "FS", "KC", "SV", "CS", "FO", "SC"]
    pitch_type = st.selectbox("Pitch Type", options=pitches)
# Pitcher handedness selection
with d1:
    sides = ["R", "L"]
    p_hand = st.selectbox("Pitcher Hand", options=sides)
# Release speed input
with c2:
    release_speed = st.number_input("Release Velocity (mph)", value=None, step=0.1)
# Spin rate input
with d2:
    spin_rate = st.number_input("Spin Rate (rpm)", value=None, step=10)
# IVB input
with c3:
    ivb = st.number_input("Induced Vertical Break (in)", value=None, step=0.1)
# HB input
with d3:
    hb = st.number_input("Horizontal Break (in)", value=None, step=0.1)
# Release height input
with c4:
    release_height = st.number_input("Release Height (ft)", value=None, step=0.1)
# Release side input
with d4:
    release_side = st.number_input("Release Side (ft)", value=None, step=0.1)
# Extension input
with c5:
    release_extension = st.number_input("Release Extension (ft)", value=None, step=0.1)
# Arm angle input
with d5:
    arm_angle = st.number_input("Arm Angle (degrees)", value=None, step=1)
# FB Velo input
with c7:
    fb_velo = st.number_input("Fastball Velo (for breaking and offspeed) (mph)", value=None, step=1)

# Create dict to store user-input values
values = {}
values["pitch_type"] = pitch_type
values["p_hand"] = p_hand
values["release_speed"] = release_speed
values["ivb"] = ivb
values["hb"] = hb
values["spin_rate"] = spin_rate
values["release_height"] = release_height
values["release_side"] = release_side
values["release_extension"] = release_extension
values["arm_angle"] = arm_angle
values["fb_velo"] = fb_velo

# Convert dict to dataframe so it can be used in our prediction function
row = pd.DataFrame([values])

# Define pitch group based on pitch type
if pitch_type in ["FF", "SI"]:
        pitch = "FB"
elif pitch_type in ["FC"]:
    pitch = "FC"
elif pitch_type in ["CH", "FS", "FO", "SC"]:
    pitch = "OFF"
elif pitch_type in ["SL", "ST", "SV", "CU", "KC", "CS"]:
    pitch = "BB"

# Define features
if pitch == "FB" or pitch == "FC":
    features = ["release_speed", "release_height", "release_side", "arm_angle", "release_extension", "ivb", "hb", "spin_rate", "plate_x", "plate_z"]
else:
    features = ["release_speed", "release_height", "release_side", "arm_angle", "release_extension", "ivb", "hb", "spin_rate", "plate_x", "plate_z", "fb_velo"]

# Create button to run program
with d7:
    button = st.button("Go")

# Read in previously calculated parquet containing run values of each pitcher / pitch type / batter side combination based on stuff and location
pitches_plus_R = pd.read_parquet("pitch_vs_R.parquet")
pitches_plus_L = pd.read_parquet("pitch_vs_L.parquet")

# Spacing
st.write("")
st.write("")
st.write("")

# Create two equal sized columns for our plots
col1, col2 = st.columns([1,1])

# When the button is pressed...
if button:

    # Plot for user-input pitch combination vs LHB on left side of page
    with col1:
        platex_range, platez_range, df_grid, grid_shape = generate_grid(row, pitch)
        xrv_grid = predict_loc_grid(df_grid, grid_shape, pitch, p_hand, "L", features, pitches_plus_L)
        fig = plot_loc_landscape(platex_range, platez_range, xrv_grid, pitch, pitch_type, p_hand, "L")
        st.image(fig_to_buf(fig))
    
    # Plot for user-input pitch combination vs RHB on right side of page
    with col2:
        platex_range, platez_range, df_grid, grid_shape = generate_grid(row, pitch)
        xrv_grid = predict_loc_grid(df_grid, grid_shape, pitch, p_hand, "R", features, pitches_plus_R)
        fig = plot_loc_landscape(platex_range, platez_range, xrv_grid, pitch, pitch_type, p_hand, "R")
        st.image(fig_to_buf(fig))
    
# Author (with LinkedIn attached)
st.markdown("## [Created By Will Friel](https://www.linkedin.com/in/william-friel/)", unsafe_allow_html=True)












    
