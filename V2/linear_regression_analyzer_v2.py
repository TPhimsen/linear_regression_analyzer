"""
Advanced Regression and Data Exclusion Analysis Application
============================================================
A comprehensive tool for mechanical alignment analysis and statistical evaluation

Author: Thanapong Phimsen
Position: Accelerator Physicist
Section: Accelerator Development Section
Division: SPS-II Technology Development Division
Organization: Synchrotron Light Research Institute (SLRI), Thailand

Date: December 2025
Version: 2.0 Enhanced (Consolidated Interface with Data Exclusion & Full Feature Set)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats
import io

# Page configuration
st.set_page_config(
    page_title="Advanced Regression Analyzer V2.0 Enhanced",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for better formatting
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stat-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .equation {
        font-size: 1.3rem;
        font-weight: 600;
        color: #2c3e50;
        text-align: center;
        padding: 1rem;
        background-color: #e8f4f8;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .highlight-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.3rem;
    }
    </style>
""", unsafe_allow_html=True)

# --- Initialize Session State ---
if 'data_points_full' not in st.session_state:
    np.random.seed(42)
    x_vals = np.sort(np.random.uniform(0, 100, 5))
    y_vals = 2.5 * x_vals + 10 + np.random.normal(0, 5, 5)
    st.session_state.data_points_full = pd.DataFrame({
        'Longitudinal Position (X)': x_vals.round(2),
        'Value (Y)': y_vals.round(2),
        'Include in Fit': [True] * 5
    })

if 'regression_results' not in st.session_state:
    st.session_state.regression_results = None

if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = pd.DataFrame()

if 'last_editor_hash' not in st.session_state:
    st.session_state.last_editor_hash = None

# Application Header
st.markdown('<p class="main-header">📊 Advanced Linear Regression Analyzer V2.0 Enhanced</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Consolidated Interface with Data Exclusion, Bulk Prediction & Enhanced Robustness</p>', unsafe_allow_html=True)

st.markdown("---")

# Sidebar for customization
with st.sidebar:
    st.header("⚙️ Configuration")
    
    position_unit = st.text_input(
        "Longitudinal Position Unit",
        value="mm",
        help="Define the unit for the X-axis (e.g., mm, m, µm)"
    )
    
    value_name = st.text_input(
        "Value Name (Y-axis)",
        value="Measured Value",
        help="Define the name/description for the Y-axis measurement"
    )
    
    st.markdown("---")
    st.markdown("### 📋 Application Info")
    st.info("""
    **V2.0 Enhanced Features:**
    - **Data Exclusion:** Toggle points for fitting
    - **Consolidated Tabs:** Streamlined workflow
    - **Robust Input:** Enhanced error handling
    - **Bulk Prediction:** Multi-point analysis
    - **Full Visualization:** Interactive plots
    """)
    
    st.markdown("---")
    st.markdown("### 👨‍🔬 Developer")
    st.markdown("""
    **Thanapong Phimsen**  
    Accelerator Physicist
    
    Accelerator Development Section  
    SPS-II Technology Development Division  
    **SLRI**, Thailand
    """)

# --- Main Content Area - CONSOLIDATED TABS ---
tab_prep, tab_analysis = st.tabs([
    "📥 Data Input & Preparation",
    "🔬 Analysis & Results"
])

# ============================================================================
# TAB 1: Data Input & Preparation
# ============================================================================
with tab_prep:
    st.header("1. 🛠️ Prepare Measured Data")
    
    col_paste, col_upload = st.columns(2)
    
    # --- Input Method 1: Paste/Manual Entry ---
    with col_paste:
        st.subheader("📋 Paste Data")
        st.markdown("""
        **Instructions:** Paste X, Y data below.  
        Supported delimiters: comma, tab, or space.
        """)
        
        text_data = st.text_area(
            "Paste data here:",
            height=200,
            placeholder="10.5, 25.3\n20.8, 52.1\n30.2, 75.8\n...",
            help="Paste data with X and Y values separated by comma, tab, or space"
        )
        
        if st.button("📤 Load Data from Text", type="primary", use_container_width=True):
            try:
                # Enhanced parsing logic from v1.2
                lines = text_data.strip().split('\n')
                data_list = []
                parse_errors = []
                
                for line_num, line in enumerate(lines, 1):
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue
                    
                    try:
                        # Enhanced delimiter handling - Priority: comma > tab > whitespace
                        if ',' in line:
                            parts = [p.strip() for p in line.split(',') if p.strip()]
                        elif '\t' in line:
                            parts = [p.strip() for p in line.split('\t') if p.strip()]
                        else:
                            parts = line.split()
                        
                        if len(parts) >= 2:
                            x_val = float(parts[0])
                            y_val = float(parts[1])
                            data_list.append([x_val, y_val, True])  # Include by default
                        else:
                            parse_errors.append(f"Line {line_num}: Insufficient values")
                            
                    except ValueError:
                        parse_errors.append(f"Line {line_num}: Invalid numeric format")
                    except Exception as e:
                        parse_errors.append(f"Line {line_num}: {str(e)}")
                
                # Display results
                if data_list:
                    new_df = pd.DataFrame(
                        data_list,
                        columns=['Longitudinal Position (X)', 'Value (Y)', 'Include in Fit']
                    )
                    st.session_state.data_points_full = new_df
                    
                    success_msg = f"✅ Successfully loaded {len(data_list)} data points!"
                    if parse_errors:
                        success_msg += f" (Skipped {len(parse_errors)} problematic lines)"
                    st.success(success_msg)
                    
                    # Show parse errors if any
                    if parse_errors:
                        with st.expander(f"⚠️ View {len(parse_errors)} parsing warnings"):
                            for error in parse_errors[:10]:
                                st.warning(error)
                            if len(parse_errors) > 10:
                                st.info(f"... and {len(parse_errors) - 10} more warnings")
                else:
                    st.error("❌ No valid data found. Please check your format.")
                    if parse_errors:
                        with st.expander("View all errors"):
                            for error in parse_errors:
                                st.error(error)
                                
            except Exception as e:
                st.error(f"❌ Unexpected error parsing data: {str(e)}")

    # --- Input Method 2: File Upload ---
    with col_upload:
        st.subheader("📁 Upload File")
        st.markdown("""
        **Instructions:** Upload CSV or TXT file.  
        First column: X, Second column: Y
        """)
        
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['csv', 'txt'],
            help="Upload CSV or TXT file with two columns"
        )
        
        if uploaded_file is not None:
            try:
                content = uploaded_file.getvalue().decode('utf-8')
                lines = content.strip().split('\n')
                
                # Enhanced delimiter detection - sample first 5 lines
                sample_lines = lines[:min(5, len(lines))]
                delimiter_scores = {',': 0, '\t': 0, 'whitespace': 0}
                
                for sample_line in sample_lines:
                    if sample_line.strip():
                        if ',' in sample_line:
                            delimiter_scores[','] += sample_line.count(',')
                        if '\t' in sample_line:
                            delimiter_scores['\t'] += sample_line.count('\t')
                        if '  ' in sample_line or (not ',' in sample_line and not '\t' in sample_line):
                            delimiter_scores['whitespace'] += 1
                
                # Select delimiter with highest score
                if delimiter_scores[','] > 0:
                    delimiter = ','
                    delimiter_name = "comma"
                elif delimiter_scores['\t'] > 0:
                    delimiter = '\t'
                    delimiter_name = "tab"
                else:
                    delimiter = r'\s+'
                    delimiter_name = "whitespace"
                
                st.info(f"📍 Detected delimiter: **{delimiter_name}**")
                
                # Read the data
                df_upload = pd.read_csv(
                    io.StringIO(content),
                    delimiter=delimiter,
                    header=None,
                    names=['Longitudinal Position (X)', 'Value (Y)'],
                    engine='python'
                )
                
                # Clean data
                df_upload = df_upload.apply(pd.to_numeric, errors='coerce').dropna()
                df_upload['Include in Fit'] = True
                
                if len(df_upload) > 0:
                    st.success(f"✅ File uploaded successfully! Found {len(df_upload)} data points.")
                    st.dataframe(df_upload[['Longitudinal Position (X)', 'Value (Y)']].head(5), use_container_width=True)
                    
                    if st.button("📤 Load Data from File", type="primary", use_container_width=True):
                        st.session_state.data_points_full = df_upload.copy()
                        st.success("✅ Data loaded into analyzer!")
                        st.rerun()
                else:
                    st.error("❌ No valid numeric data found in file after parsing.")
                    
            except UnicodeDecodeError:
                st.error("❌ Error: File encoding not supported. Please use UTF-8 encoded files.")
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
                st.info("💡 Tip: Ensure your file has two columns (X and Y) separated by comma, tab, or spaces.")

    st.markdown("---")

    # --- Interactive Data Editor with Exclusion Control ---
    st.header("2. ✏️ Interactive Data Editor & Exclusion Control")
    
    # Display current status
    df_current = st.session_state.data_points_full
    if len(df_current) > 0:
        included_count = df_current['Include in Fit'].sum()
        excluded_count = len(df_current) - included_count
        
        col_status1, col_status2, col_status3 = st.columns(3)
        with col_status1:
            st.metric("📊 Total Points", len(df_current))
        with col_status2:
            st.metric("✅ Included in Fit", included_count, delta=None, delta_color="normal")
        with col_status3:
            st.metric("❌ Excluded from Fit", excluded_count, delta=None, delta_color="inverse")
    
    st.markdown("""
    **Instructions:**
    - Edit values directly in the table
    - Add/remove rows using table controls
    - **Uncheck 'Include in Fit'** to exclude a point from regression (it remains visible on plots)
    - Changes are automatically validated and saved
    """)
    
    # Column configuration for better UX
    column_config = {
        "Include in Fit": st.column_config.CheckboxColumn(
            "Include in Fit",
            default=True,
            help="Uncheck to exclude this point from the regression calculation",
            width="medium"
        ),
        'Longitudinal Position (X)': st.column_config.NumberColumn(
            f"Position ({position_unit})",
            format="%.4f",
            width="medium"
        ),
        'Value (Y)': st.column_config.NumberColumn(
            f"{value_name}",
            format="%.4f",
            width="medium"
        )
    }
    
    # Display editable dataframe
    edited_df = st.data_editor(
        st.session_state.data_points_full,
        num_rows="dynamic",
        use_container_width=True,
        column_config=column_config,
        hide_index=True,
        key="data_editor_v2_enhanced"
    )
    
    # Immediate validation and persistence with loop prevention
    try:
        # Create hash to detect actual changes
        current_hash = hash(str(edited_df.values.tobytes()))
        
        if current_hash != st.session_state.last_editor_hash:
            # Validate and clean data
            edited_df_clean = edited_df.copy()
            
            # Ensure numeric columns
            for col in ['Longitudinal Position (X)', 'Value (Y)']:
                edited_df_clean[col] = pd.to_numeric(edited_df_clean[col], errors='coerce')
            
            # Ensure boolean column
            edited_df_clean['Include in Fit'] = edited_df_clean['Include in Fit'].fillna(True).astype(bool)
            
            # Drop rows with invalid X or Y
            edited_df_clean = edited_df_clean.dropna(subset=['Longitudinal Position (X)', 'Value (Y)'])
            
            if len(edited_df_clean) > 0:
                st.session_state.data_points_full = edited_df_clean
                st.session_state.last_editor_hash = current_hash
                
                included_count = edited_df_clean['Include in Fit'].sum()
                st.info(f"💾 **Auto-saved:** {len(edited_df_clean)} total points, {included_count} included in next fit")
            else:
                st.warning("⚠️ No valid numeric data in the editor. Please enter valid numbers.")
                
    except Exception as e:
        st.error(f"❌ Error validating data: {str(e)}")
    
    st.markdown("---")
    
    # Data generation tools
    st.markdown("### 🛠️ Data Generation Tools")
    col_tools1, col_tools2 = st.columns(2)
    
    with col_tools1:
        if st.button("🎲 Generate New Random Data", use_container_width=True):
            np.random.seed(None)
            x_vals = np.sort(np.random.uniform(0, 100, 5))
            y_vals = 2.5 * x_vals + 10 + np.random.normal(0, 5, 5)
            st.session_state.data_points_full = pd.DataFrame({
                'Longitudinal Position (X)': x_vals.round(2),
                'Value (Y)': y_vals.round(2),
                'Include in Fit': [True] * 5
            })
            st.session_state.regression_results = None
            st.session_state.prediction_results = pd.DataFrame()
            st.rerun()
    
    with col_tools2:
        if st.button("🗑️ Clear All Data", use_container_width=True):
            st.session_state.data_points_full = pd.DataFrame(
                columns=['Longitudinal Position (X)', 'Value (Y)', 'Include in Fit']
            )
            st.session_state.regression_results = None
            st.session_state.prediction_results = pd.DataFrame()
            st.rerun()

# ============================================================================
# TAB 2: Analysis & Results
# ============================================================================
with tab_analysis:
    st.header("3. 🔬 Regression Analysis")
    
    # Get full dataset and filter for included points
    df_full = st.session_state.data_points_full
    df_included = df_full[df_full['Include in Fit']].copy()
    
    # Pre-analysis validation and status
    if len(df_full) == 0:
        st.warning("⚠️ **No Data:** Please load data in the 'Data Input & Preparation' tab first.")
        st.stop()
    
    # Display dataset status
    included_count = len(df_included)
    excluded_count = len(df_full) - included_count
    
    col_pre1, col_pre2, col_pre3, col_pre4 = st.columns(4)
    with col_pre1:
        st.metric("📊 Total Points", len(df_full))
    with col_pre2:
        st.metric("✅ Included", included_count)
    with col_pre3:
        st.metric("❌ Excluded", excluded_count)
    with col_pre4:
        if included_count >= 2:
            st.metric("Status", "Ready", delta="✓", delta_color="normal")
        else:
            st.metric("Status", "Need Data", delta="✗", delta_color="inverse")
    
    st.markdown("---")
    
    # Validation checks
    if included_count < 2:
        st.error(f"❌ **Insufficient Data:** Only {included_count} point(s) marked 'Include in Fit'. At least 2 points required for linear regression.")
        st.info("💡 **Solution:** Go to the 'Data Input & Preparation' tab and ensure at least 2 points have 'Include in Fit' checked.")
        st.stop()
    
    # Calculate button
    if st.button("🔬 Calculate Regression Analysis", type="primary", use_container_width=True):
        
        X = df_included['Longitudinal Position (X)'].values
        Y = df_included['Value (Y)'].values
        
        # Validate X variance
        x_variance = np.var(X)
        if x_variance == 0 or np.allclose(X, X[0]):
            st.error("❌ **Analysis Error:** All X values in the INCLUDED dataset are identical. Linear regression requires variation in X positions.")
            st.warning("💡 **Solution:** Please include points with different longitudinal positions.")
            st.stop()
        
        # Perform linear regression on included points only
        slope, intercept, r_value, p_value, std_err_slope = stats.linregress(X, Y)
        
        # Calculate statistics
        r_squared = r_value ** 2
        tilt_angle_deg = np.degrees(np.arctan(slope))
        Y_pred = slope * X + intercept
        deviations = Y - Y_pred
        n = len(X)
        residuals_squared = deviations ** 2
        std_err_estimate = np.sqrt(np.sum(residuals_squared) / (n - 2)) if n > 2 else 0
        
        # Store results
        st.session_state.regression_results = {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_squared,
            'std_err_slope': std_err_slope,
            'std_err_estimate': std_err_estimate,
            'tilt_angle': tilt_angle_deg,
            'X': X,
            'Y': Y,
            'Y_pred': Y_pred,
            'deviations': deviations,
            'n_points': n
        }
        
        st.success(f"✅ Regression analysis completed using {n} included points!")
        st.rerun()

    # Display results if available
    if st.session_state.regression_results is not None:
        results = st.session_state.regression_results
        
        st.markdown("---")
        st.markdown("### 📈 Statistical Results")
        
        # Highlight box showing fit basis
        st.markdown(
            f'<div class="highlight-box">📌 <strong>Regression calculated using {results["n_points"]} included points</strong> '
            f'(Excluded points shown on plot for reference but not used in calculation)</div>',
            unsafe_allow_html=True
        )
        
        # Trendline equation
        eq_sign = '+' if results['intercept'] >= 0 else ''
        st.markdown(
            f'<div class="equation">Y = {results["slope"]:.6f} × X {eq_sign} {results["intercept"]:.6f}</div>',
            unsafe_allow_html=True
        )
        
        # Statistics display
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="stat-box">', unsafe_allow_html=True)
            st.metric("Slope (m)", f"{results['slope']:.6f}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="stat-box">', unsafe_allow_html=True)
            st.metric("Tilt Angle (θ)", f"{results['tilt_angle']:.4f}°")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="stat-box">', unsafe_allow_html=True)
            st.metric("Intercept (c)", f"{results['intercept']:.6f}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="stat-box">', unsafe_allow_html=True)
            st.metric("R² (Fit Quality)", f"{results['r_squared']:.6f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="stat-box">', unsafe_allow_html=True)
            st.metric("Std Error (Slope)", f"{results['std_err_slope']:.6f}",
                     help="Standard error of the slope coefficient")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="stat-box">', unsafe_allow_html=True)
            st.metric("Std Error (Estimate)", f"{results['std_err_estimate']:.6f}",
                     help="Standard error of the estimate (residual standard deviation)")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Additional info row
        st.markdown("---")
        col_info1, col_info2, col_info3 = st.columns(3)
        with col_info1:
            st.markdown('<div class="stat-box">', unsafe_allow_html=True)
            st.metric("Data Points (n)", results['n_points'])
            st.markdown('</div>', unsafe_allow_html=True)
        with col_info2:
            st.markdown('<div class="stat-box">', unsafe_allow_html=True)
            degrees_freedom = results['n_points'] - 2
            st.metric("Degrees of Freedom", degrees_freedom,
                     help="n - 2 for linear regression")
            st.markdown('</div>', unsafe_allow_html=True)
        with col_info3:
            st.markdown('<div class="stat-box">', unsafe_allow_html=True)
            st.metric("Excluded Points", len(df_full) - results['n_points'],
                     help="Points not used in regression calculation")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Visualization
        st.markdown("### 📉 Visualization")
        
        # Prepare plot data - calculate predictions for ALL points
        df_plot = df_full.copy()
        df_plot['Predicted Y'] = results['slope'] * df_plot['Longitudinal Position (X)'] + results['intercept']
        df_plot['Deviation'] = df_plot['Value (Y)'] - df_plot['Predicted Y']
        
        # Create figure
        fig = go.Figure()
        
        # Add excluded points (if any)
        df_excluded_plot = df_plot[~df_plot['Include in Fit']]
        if not df_excluded_plot.empty:
            fig.add_trace(go.Scatter(
                x=df_excluded_plot['Longitudinal Position (X)'],
                y=df_excluded_plot['Value (Y)'],
                mode='markers',
                name=f'Excluded (n={len(df_excluded_plot)})',
                marker=dict(
                    size=10,
                    color='#95a5a6',
                    symbol='x',
                    line=dict(width=2, color='#7f8c8d')
                ),
                hovertemplate=(
                    f'<b>Status:</b> Excluded<br>'
                    f'Position: %{{x:.2f}} {position_unit}<br>'
                    f'{value_name}: %{{y:.2f}}<br>'
                    f'Deviation: %{{customdata[0]:.4f}}'
                    f'<extra></extra>'
                ),
                customdata=np.stack((df_excluded_plot['Deviation'],), axis=-1)
            ))
        
        # Add included points
        df_included_plot = df_plot[df_plot['Include in Fit']]
        fig.add_trace(go.Scatter(
            x=df_included_plot['Longitudinal Position (X)'],
            y=df_included_plot['Value (Y)'],
            mode='markers',
            name=f'Included (n={len(df_included_plot)})',
            marker=dict(
                size=10,
                color='#e74c3c',
                symbol='circle',
                line=dict(width=1, color='darkred')
            ),
            hovertemplate=(
                f'<b>Status:</b> Included<br>'
                f'Position: %{{x:.2f}} {position_unit}<br>'
                f'{value_name}: %{{y:.2f}}<br>'
                f'Deviation: %{{customdata[0]:.4f}}'
                f'<extra></extra>'
            ),
            customdata=np.stack((df_included_plot['Deviation'],), axis=-1)
        ))
        
        # Add trendline
        X_line = np.linspace(
            df_plot['Longitudinal Position (X)'].min(),
            df_plot['Longitudinal Position (X)'].max(),
            100
        )
        Y_line = results['slope'] * X_line + results['intercept']
        
        fig.add_trace(go.Scatter(
            x=X_line,
            y=Y_line,
            mode='lines',
            name=f'Linear Fit (R²={results["r_squared"]:.4f})',
            line=dict(color='#3498db', width=3, dash='solid'),
            hovertemplate=f'Position: %{{x:.2f}} {position_unit}<br>Predicted: %{{y:.2f}}<extra></extra>'
        ))
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f'Linear Regression: {value_name} vs Longitudinal Position',
                font=dict(size=18, color='#2c3e50')
            ),
            xaxis_title=f'Longitudinal Position ({position_unit})',
            yaxis_title=f'{value_name}',
            hovermode='closest',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            template='plotly_white',
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Deviation Table
        st.markdown("### 📊 Deviation Analysis")
        
        # Create comprehensive deviation table
        deviation_df = df_plot.copy()
        deviation_df = deviation_df.rename(columns={
            'Longitudinal Position (X)': f'Position ({position_unit})',
            'Value (Y)': f'{value_name} (Measured)',
            'Predicted Y': f'{value_name} (Predicted)',
            'Include in Fit': 'Used in Fit'
        })
        
        # Reorder columns
        deviation_df = deviation_df[[
            'Used in Fit',
            f'Position ({position_unit})',
            f'{value_name} (Measured)',
            f'{value_name} (Predicted)',
            'Deviation'
        ]]
        
        st.dataframe(
            deviation_df.style.format({
                f'Position ({position_unit})': '{:.4f}',
                f'{value_name} (Measured)': '{:.4f}',
                f'{value_name} (Predicted)': '{:.4f}',
                'Deviation': '{:.4f}'
            }),
            use_container_width=True
        )
        
        # Download button for deviation table
        csv_deviation = deviation_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Deviation Table (CSV)",
            data=csv_deviation,
            file_name="regression_deviation_analysis_v2.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        st.markdown("---")
        
        # Prediction Tool
        st.markdown("### 🎯 Prediction Tool")
        st.markdown("Enter one or more longitudinal positions to predict the corresponding values.")
        
        col_input1, col_input2 = st.columns([3, 2])
        
        with col_input1:
            st.markdown("**📋 Bulk Prediction Input**")
            prediction_text = st.text_area(
                "Enter positions (one per line):",
                value="",
                height=150,
                placeholder="10.0\n25.5\n50.0\n75.5\n99.9\n...",
                help="Enter one position value per line for bulk prediction"
            )
            
            if st.button("🔮 Calculate Predictions", type="primary", use_container_width=True):
                try:
                    # Enhanced parsing with error handling
                    lines = prediction_text.strip().split('\n')
                    x_new_values = []
                    parse_errors_pred = []
                    
                    for line_num, line in enumerate(lines, 1):
                        line = line.strip()
                        if not line:  # Skip empty lines
                            continue
                        try:
                            x_val = float(line)
                            x_new_values.append(x_val)
                        except ValueError:
                            parse_errors_pred.append(f"Line {line_num}: '{line}' is not a valid number")
                    
                    if x_new_values:
                        # Calculate predictions
                        x_new_array = np.array(x_new_values)
                        y_predicted_array = results['slope'] * x_new_array + results['intercept']
                        
                        # Store in session state
                        st.session_state.prediction_results = pd.DataFrame({
                            f'Longitudinal Position ({position_unit})': x_new_array,
                            f'Predicted {value_name}': y_predicted_array
                        })
                        
                        st.success(f"✅ Calculated {len(x_new_values)} predictions!")
                        
                        if parse_errors_pred:
                            with st.expander(f"⚠️ Skipped {len(parse_errors_pred)} invalid lines"):
                                for error in parse_errors_pred:
                                    st.warning(error)
                    else:
                        st.error("❌ No valid position values found. Please enter numeric values.")
                        if parse_errors_pred:
                            with st.expander("View all errors"):
                                for error in parse_errors_pred:
                                    st.error(error)
                                    
                except Exception as e:
                    st.error(f"❌ Error processing predictions: {str(e)}")
        
        with col_input2:
            st.markdown("**⚡ Quick Single Prediction**")
            x_single = st.number_input(
                f"Position ({position_unit}):",
                value=float(results['X'].mean()),
                format="%.4f",
                help="Enter a single position for quick prediction"
            )
            
            if st.button("Calculate Single", use_container_width=True):
                y_single = results['slope'] * x_single + results['intercept']
                
                # Store as single-row dataframe
                st.session_state.prediction_results = pd.DataFrame({
                    f'Longitudinal Position ({position_unit})': [x_single],
                    f'Predicted {value_name}': [y_single]
                })
                st.success("✅ Single prediction calculated!")
        
        # Display prediction results if available
        if not st.session_state.prediction_results.empty:
            st.markdown("---")
            st.markdown("#### 📊 Prediction Results")
            
            pred_df = st.session_state.prediction_results
            
            # Format and display the table
            st.dataframe(
                pred_df.style.format({
                    f'Longitudinal Position ({position_unit})': '{:.4f}',
                    f'Predicted {value_name}': '{:.6f}'
                }),
                use_container_width=True
            )
            
            # Summary statistics
            if len(pred_df) > 0:
                col_stats1, col_stats2, col_stats3 = st.columns(3)
                with col_stats1:
                    st.metric("Total Predictions", len(pred_df))
                with col_stats2:
                    st.metric("Min Predicted Value", f"{pred_df[f'Predicted {value_name}'].min():.4f}")
                with col_stats3:
                    st.metric("Max Predicted Value", f"{pred_df[f'Predicted {value_name}'].max():.4f}")
            
            # Download button
            csv_pred = pred_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Prediction Results (CSV)",
                data=csv_pred,
                file_name="regression_predictions_v2.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            # Visualization option
            st.markdown("---")
            show_predictions = st.checkbox(
                "📈 Show predictions on graph",
                value=False,
                help="Overlay predicted points on the regression plot"
            )
            
            if show_predictions:
                # Create enhanced figure with predictions
                fig_pred = go.Figure()
                
                # Add excluded points (if any)
                if not df_excluded_plot.empty:
                    fig_pred.add_trace(go.Scatter(
                        x=df_excluded_plot['Longitudinal Position (X)'],
                        y=df_excluded_plot['Value (Y)'],
                        mode='markers',
                        name=f'Excluded Data (n={len(df_excluded_plot)})',
                        marker=dict(
                            size=10,
                            color='#95a5a6',
                            symbol='x',
                            line=dict(width=2, color='#7f8c8d')
                        ),
                        hovertemplate=f'Status: Excluded<br>Position: %{{x:.2f}} {position_unit}<br>{value_name}: %{{y:.2f}}<extra></extra>'
                    ))
                
                # Add included (measured) data
                fig_pred.add_trace(go.Scatter(
                    x=df_included_plot['Longitudinal Position (X)'],
                    y=df_included_plot['Value (Y)'],
                    mode='markers',
                    name=f'Measured Data (n={len(df_included_plot)})',
                    marker=dict(
                        size=10,
                        color='#e74c3c',
                        symbol='circle',
                        line=dict(width=1, color='darkred')
                    ),
                    hovertemplate=f'Status: Measured<br>Position: %{{x:.2f}} {position_unit}<br>{value_name}: %{{y:.2f}}<extra></extra>'
                ))
                
                # Add trendline - extend to cover prediction range
                X_line = np.linspace(
                    min(df_plot['Longitudinal Position (X)'].min(), pred_df[f'Longitudinal Position ({position_unit})'].min()),
                    max(df_plot['Longitudinal Position (X)'].max(), pred_df[f'Longitudinal Position ({position_unit})'].max()),
                    100
                )
                Y_line = results['slope'] * X_line + results['intercept']
                
                fig_pred.add_trace(go.Scatter(
                    x=X_line,
                    y=Y_line,
                    mode='lines',
                    name='Linear Fit',
                    line=dict(color='#3498db', width=3, dash='solid'),
                    hovertemplate=f'Position: %{{x:.2f}} {position_unit}<br>Predicted: %{{y:.2f}}<extra></extra>'
                ))
                
                # Add prediction points
                fig_pred.add_trace(go.Scatter(
                    x=pred_df[f'Longitudinal Position ({position_unit})'],
                    y=pred_df[f'Predicted {value_name}'],
                    mode='markers',
                    name=f'Predictions (n={len(pred_df)})',
                    marker=dict(
                        size=12,
                        color='#2ecc71',
                        symbol='star',
                        line=dict(width=2, color='darkgreen')
                    ),
                    hovertemplate=f'Predicted Position: %{{x:.2f}} {position_unit}<br>Predicted {value_name}: %{{y:.2f}}<extra></extra>'
                ))
                
                # Update layout
                fig_pred.update_layout(
                    title=dict(
                        text=f'Linear Regression with {len(pred_df)} Prediction Point(s)',
                        font=dict(size=18, color='#2c3e50')
                    ),
                    xaxis_title=f'Longitudinal Position ({position_unit})',
                    yaxis_title=f'{value_name}',
                    hovermode='closest',
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    template='plotly_white',
                    height=600
                )
                
                st.plotly_chart(fig_pred, use_container_width=True)
                
                # Additional analysis for multiple predictions
                if len(pred_df) > 1:
                    st.markdown("---")
                    st.markdown("#### 📈 Prediction Range Analysis")
                    
                    pred_range = pred_df[f'Predicted {value_name}'].max() - pred_df[f'Predicted {value_name}'].min()
                    position_range = pred_df[f'Longitudinal Position ({position_unit})'].max() - pred_df[f'Longitudinal Position ({position_unit})'].min()
                    
                    col_range1, col_range2 = st.columns(2)
                    with col_range1:
                        st.metric("Position Range", f"{position_range:.4f} {position_unit}")
                    with col_range2:
                        st.metric("Predicted Value Range", f"{pred_range:.6f}")
            
        else:
            st.info("💡 Enter position values above and click 'Calculate Predictions' to see results.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d; padding: 1rem;'>
    <p><strong>Advanced Linear Regression Analyzer</strong> | Version 2.0 Enhanced</p>
    <p>Developed by: <strong>Thanapong Phimsen</strong>, Accelerator Physicist</p>
    <p>Accelerator Development Section, SPS-II Technology Development Division</p>
    <p><strong>Synchrotron Light Research Institute (SLRI)</strong>, Thailand</p>
    <p style='margin-top: 0.5rem; font-size: 0.9rem;'>For precision engineering and scientific analysis | December 2025</p>
</div>
""", unsafe_allow_html=True)
