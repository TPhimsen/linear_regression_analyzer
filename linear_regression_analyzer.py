"""
Advanced Linear Regression Analysis Application
================================================
A comprehensive tool for mechanical alignment analysis and statistical evaluation

Author: Thanapong Phimsen
Position: Accelerator Physicist
Section: Accelerator Development Section
Division: SPS-II Technology Development Division
Organization: Synchrotron Light Research Institute (SLRI), Thailand

Date: December 2025
Version: 1.2
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import io

# Page configuration
st.set_page_config(
    page_title="Advanced Linear Regression Analyzer",
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
    </style>
""", unsafe_allow_html=True)

# Initialize session state for data persistence
if 'data_points' not in st.session_state:
    # Generate 5 realistic random data points for mechanical alignment
    np.random.seed(42)
    x_vals = np.sort(np.random.uniform(0, 100, 5))
    # Linear relationship with some noise
    y_vals = 2.5 * x_vals + 10 + np.random.normal(0, 5, 5)
    st.session_state.data_points = pd.DataFrame({
        'Longitudinal Position (X)': x_vals.round(2),
        'Value (Y)': y_vals.round(2)
    })

# Application Header
st.markdown('<p class="main-header">📊 Advanced Linear Regression Analyzer</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Precision Analysis Tool for Mechanical Alignment & Statistical Evaluation</p>', unsafe_allow_html=True)

st.markdown("---")

# Sidebar for customization
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Unit customization
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
    **Features:**
    - Multiple data input methods
    - Real-time linear regression
    - Comprehensive statistics
    - Interactive visualization
    - Prediction functionality
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


# Main content area
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Paste/Manual Entry",
    "📁 Upload File",
    "✏️ Interactive Editor",
    "📊 Analysis & Results"
])

# Tab 1: Paste from Excel / Manual Entry
with tab1:
    st.header("📋 Paste Data from Excel or Manual Entry")
    st.markdown("""
    **Instructions:** Paste your data below. Each row should contain X and Y values separated by comma, tab, or space.
    
    Example formats:
    - `10.5, 25.3`
    - `10.5    25.3`
    - `10.5 25.3`
    """)
    
    text_data = st.text_area(
        "Paste your data here:",
        height=200,
        placeholder="10.5, 25.3\n20.8, 52.1\n30.2, 75.8\n..."
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("Load Data from Text", type="primary"):
            try:
                # Parse the text data with enhanced robustness
                lines = text_data.strip().split('\n')
                data_list = []
                parse_errors = []
                
                for line_num, line in enumerate(lines, 1):
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue
                    
                    try:
                        # Enhanced delimiter handling
                        # Priority: comma > tab > whitespace
                        if ',' in line:
                            # Split by comma, then clean up any extra whitespace
                            parts = [p.strip() for p in line.split(',') if p.strip()]
                        elif '\t' in line:
                            # Split by tab, then clean up any extra whitespace
                            parts = [p.strip() for p in line.split('\t') if p.strip()]
                        else:
                            # Split by any whitespace (handles multiple spaces)
                            parts = line.split()
                        
                        # Validate we have at least 2 numeric values
                        if len(parts) >= 2:
                            # Try to convert to float, handling any remaining whitespace
                            x_val = float(parts[0])
                            y_val = float(parts[1])
                            data_list.append([x_val, y_val])
                        else:
                            parse_errors.append(f"Line {line_num}: Insufficient values")
                            
                    except ValueError as ve:
                        parse_errors.append(f"Line {line_num}: Invalid numeric format")
                    except Exception as e:
                        parse_errors.append(f"Line {line_num}: {str(e)}")
                
                # Display results
                if data_list:
                    st.session_state.data_points = pd.DataFrame(
                        data_list,
                        columns=['Longitudinal Position (X)', 'Value (Y)']
                    )
                    success_msg = f"✅ Successfully loaded {len(data_list)} data points!"
                    if parse_errors:
                        success_msg += f"\n\n⚠️ Skipped {len(parse_errors)} problematic lines"
                    st.success(success_msg)
                    
                    # Show parse errors in expander if any
                    if parse_errors:
                        with st.expander("View parsing warnings"):
                            for error in parse_errors[:10]:  # Limit to first 10 errors
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

# Tab 2: Upload File
with tab2:
    st.header("📁 Upload Data File")
    st.markdown("""
    **Instructions:** Upload a CSV or TXT file containing your data.
    
    File format requirements:
    - First column: Longitudinal Position (X)
    - Second column: Value (Y)
    - Header row is optional
    """)
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['csv', 'txt'],
        help="Upload CSV or TXT file with X and Y data"
    )
    
    if uploaded_file is not None:
        try:
            # Try to read the file with enhanced delimiter detection
            content = uploaded_file.getvalue().decode('utf-8')
            lines = content.strip().split('\n')
            
            # Enhanced delimiter detection - sample first 5 lines (or all if fewer)
            sample_lines = lines[:min(5, len(lines))]
            delimiter_scores = {',': 0, '\t': 0, 'whitespace': 0}
            
            for sample_line in sample_lines:
                if sample_line.strip():
                    # Count occurrences of each delimiter
                    if ',' in sample_line:
                        delimiter_scores[','] += sample_line.count(',')
                    if '\t' in sample_line:
                        delimiter_scores['\t'] += sample_line.count('\t')
                    # Check for multiple spaces (likely whitespace-delimited)
                    if '  ' in sample_line or (not ',' in sample_line and not '\t' in sample_line):
                        delimiter_scores['whitespace'] += 1
            
            # Select delimiter with highest score (priority: comma > tab > whitespace)
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
            
            # Read the data with detected delimiter
            df_upload = pd.read_csv(
                io.StringIO(content),
                delimiter=delimiter,
                header=None,
                names=['Longitudinal Position (X)', 'Value (Y)'],
                engine='python'
            )
            
            # Remove any non-numeric rows (potential headers) and clean data
            df_upload = df_upload.apply(pd.to_numeric, errors='coerce').dropna()
            
            if len(df_upload) == 0:
                st.error("❌ No valid numeric data found in file after parsing.")
            else:
                st.success(f"✅ File uploaded successfully! Found {len(df_upload)} data points.")
                st.dataframe(df_upload, use_container_width=True)
                
                if st.button("Load Data from File", type="primary"):
                    st.session_state.data_points = df_upload.copy()
                    st.success("✅ Data loaded into analyzer!")
                    st.rerun()
                
        except UnicodeDecodeError:
            st.error("❌ Error: File encoding not supported. Please use UTF-8 encoded files.")
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
            st.info("💡 Tip: Ensure your file has two columns (X and Y) separated by comma, tab, or spaces.")

# Tab 3: Manual/Randomized Entry
with tab3:
    st.header("✏️ Interactive Data Editor")
    st.markdown("""
    **Instructions:** Edit the table below to add, remove, or modify data points.
    - Click on cells to edit values
    - Add/remove rows using the table controls
    - **Changes are automatically saved** when you interact with other parts of the app
    """)
    
    # Display editable dataframe with automatic persistence
    edited_df = st.data_editor(
        st.session_state.data_points,
        num_rows="dynamic",
        use_container_width=True,
        key="data_editor"
    )
    
    # Immediate validation and persistence
    # Clean the data: remove NaN, ensure numeric types
    try:
        edited_df_clean = edited_df.dropna()
        # Ensure columns are numeric
        edited_df_clean['Longitudinal Position (X)'] = pd.to_numeric(
            edited_df_clean['Longitudinal Position (X)'], errors='coerce'
        )
        edited_df_clean['Value (Y)'] = pd.to_numeric(
            edited_df_clean['Value (Y)'], errors='coerce'
        )
        # Remove any rows that couldn't be converted to numeric
        edited_df_clean = edited_df_clean.dropna()
        
        # Update session state immediately if data is valid
        if len(edited_df_clean) > 0:
            st.session_state.data_points = edited_df_clean
            st.info(f"📊 Current dataset: **{len(edited_df_clean)} valid data points** (automatically saved)")
        else:
            st.warning("⚠️ No valid numeric data in the editor. Please enter valid numbers.")
    except Exception as e:
        st.error(f"❌ Error validating data: {str(e)}")
    
    st.markdown("---")
    st.markdown("### 🛠️ Data Generation Tools")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🎲 Generate New Random Data", use_container_width=True):
            np.random.seed(None)  # Use current time as seed
            x_vals = np.sort(np.random.uniform(0, 100, 5))
            y_vals = 2.5 * x_vals + 10 + np.random.normal(0, 5, 5)
            st.session_state.data_points = pd.DataFrame({
                'Longitudinal Position (X)': x_vals.round(2),
                'Value (Y)': y_vals.round(2)
            })
            st.rerun()
    
    with col2:
        if st.button("🗑️ Clear All Data", use_container_width=True):
            st.session_state.data_points = pd.DataFrame({
                'Longitudinal Position (X)': [],
                'Value (Y)': []
            })
            st.rerun()

# Tab 4: Analysis & Results
with tab4:
    st.header("📊 Linear Regression Analysis & Results")
    
    # Check if we have enough data
    if len(st.session_state.data_points) < 2:
        st.warning("⚠️ Please provide at least 2 data points to perform regression analysis.")
    else:
        # Display current data
        with st.expander("📋 Current Data Points", expanded=False):
            st.dataframe(st.session_state.data_points, use_container_width=True)
        
        # Calculate button
        if st.button("🔬 Calculate Regression Analysis", type="primary", use_container_width=True):
            
            # Extract data
            X = st.session_state.data_points['Longitudinal Position (X)'].values
            Y = st.session_state.data_points['Value (Y)'].values
            
            # Validate X variance (ensure not all X values are the same)
            x_variance = np.var(X)
            if x_variance == 0 or np.allclose(X, X[0]):
                st.error("❌ **Analysis Error:** All X values are identical. Linear regression requires variation in X positions.")
                st.warning("💡 **Solution:** Please ensure your longitudinal positions have different values.")
                st.stop()
            
            # Perform linear regression
            slope, intercept, r_value, p_value, std_err_slope = stats.linregress(X, Y)
            
            # Calculate additional statistics
            r_squared = r_value ** 2
            tilt_angle_rad = np.arctan(slope)
            tilt_angle_deg = np.degrees(tilt_angle_rad)
            
            # Predictions and residuals
            Y_pred = slope * X + intercept
            deviations = Y - Y_pred
            
            # Calculate Standard Error of the Estimate (Se)
            # Se = sqrt(sum of squared residuals / (n - 2))
            n = len(X)
            residuals_squared = deviations ** 2
            std_err_estimate = np.sqrt(np.sum(residuals_squared) / (n - 2)) if n > 2 else 0
            
            # Store results in session state
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
            
            st.success("✅ Regression analysis completed successfully!")
        
        # Display results if available
        if 'regression_results' in st.session_state:
            results = st.session_state.regression_results
            
            st.markdown("---")
            st.markdown("### 📈 Statistical Results")
            
            # Display trendline equation
            eq_sign = '+' if results['intercept'] >= 0 else ''
            st.markdown(
                f'<div class="equation">Y = {results["slope"]:.6f} × X {eq_sign} {results["intercept"]:.6f}</div>',
                unsafe_allow_html=True
            )
            
            # Display statistics in columns
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
            
            st.markdown("---")
            
            # Deviation Table
            st.markdown("### 📊 Deviation Analysis")
            deviation_df = pd.DataFrame({
                f'Longitudinal Position ({position_unit})': results['X'],
                f'{value_name} (Measured)': results['Y'],
                f'{value_name} (Predicted)': results['Y_pred'],
                'Deviation (Y - Y_pred)': results['deviations']
            })
            
            # Format the dataframe
            st.dataframe(
                deviation_df.style.format({
                    f'Longitudinal Position ({position_unit})': '{:.4f}',
                    f'{value_name} (Measured)': '{:.4f}',
                    f'{value_name} (Predicted)': '{:.4f}',
                    'Deviation (Y - Y_pred)': '{:.4f}'
                }),
                use_container_width=True
            )
            
            # Download button for deviation table
            csv = deviation_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Deviation Table (CSV)",
                data=csv,
                file_name="regression_deviation_analysis.csv",
                mime="text/csv"
            )
            
            st.markdown("---")
            
            # Interactive Plot
            st.markdown("### 📉 Visualization")
            
            # Create figure
            fig = go.Figure()
            
            # Add scatter plot for original data
            fig.add_trace(go.Scatter(
                x=results['X'],
                y=results['Y'],
                mode='markers',
                name='Measured Data',
                marker=dict(
                    size=10,
                    color='#e74c3c',
                    symbol='circle',
                    line=dict(width=1, color='darkred')
                ),
                hovertemplate=f'Position: %{{x:.2f}} {position_unit}<br>{value_name}: %{{y:.2f}}<extra></extra>'
            ))
            
            # Add trendline
            X_line = np.linspace(results['X'].min(), results['X'].max(), 100)
            Y_line = results['slope'] * X_line + results['intercept']
            
            fig.add_trace(go.Scatter(
                x=X_line,
                y=Y_line,
                mode='lines',
                name='Linear Fit',
                line=dict(color='#3498db', width=3, dash='solid'),
                hovertemplate=f'Position: %{{x:.2f}} {position_unit}<br>Predicted: %{{y:.2f}}<extra></extra>'
            ))
            
            # Update layout
            fig.update_layout(
                title=dict(
                    text=f'Linear Regression Analysis: {value_name} vs Longitudinal Position',
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
            
            # Prediction Section
            st.markdown("### 🎯 Prediction Tool")
            st.markdown("Enter one or more longitudinal positions to predict the corresponding values. You can enter multiple positions, one per line.")
            
            # Create two columns for input methods
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
                    # Parse the input
                    try:
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
            if 'prediction_results' in st.session_state and not st.session_state.prediction_results.empty:
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
                    file_name="regression_predictions.csv",
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
                    
                    # Add original scatter plot
                    fig_pred.add_trace(go.Scatter(
                        x=results['X'],
                        y=results['Y'],
                        mode='markers',
                        name='Measured Data',
                        marker=dict(
                            size=10,
                            color='#e74c3c',
                            symbol='circle',
                            line=dict(width=1, color='darkred')
                        ),
                        hovertemplate=f'Position: %{{x:.2f}} {position_unit}<br>{value_name}: %{{y:.2f}}<extra></extra>'
                    ))
                    
                    # Add trendline
                    X_line = np.linspace(
                        min(results['X'].min(), pred_df[f'Longitudinal Position ({position_unit})'].min()),
                        max(results['X'].max(), pred_df[f'Longitudinal Position ({position_unit})'].max()),
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
    <p><strong>Advanced Linear Regression Analyzer</strong> | Version 1.2</p>
    <p>Developed by: <strong>Thanapong Phimsen</strong>, Accelerator Physicist</p>
    <p>Accelerator Development Section, SPS-II Technology Development Division</p>
    <p><strong>Synchrotron Light Research Institute (SLRI)</strong>, Thailand</p>
    <p style='margin-top: 0.5rem; font-size: 0.9rem;'>For precision engineering and scientific analysis | December 2025</p>
</div>
""", unsafe_allow_html=True)
