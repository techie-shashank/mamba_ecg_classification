#!/usr/bin/env python3
"""
Streamlit Dashboard for ML Experiment Tracking
Visualizes experiment data from database or CSV files
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
import json
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
from PIL import Image
import base64
import io

# Page configuration
st.set_page_config(
    page_title="ML Experiment Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)


class ExperimentDashboard:
    """Class to handle dashboard functionality"""

    def __init__(self, db_path: str = "outputs/experiments.db", csv_path: str = "outputs/experiments_data.csv"):
        """
        Initialize dashboard

        Args:
            db_path: Path to SQLite database
            csv_path: Path to CSV file
        """
        self.db_path = db_path
        self.csv_path = csv_path
        self.df = None

    def safe_format_metric(self, value, precision: int = 4, suffix: str = "") -> str:
        """
        Safely format a metric value, handling None values
        
        Args:
            value: The value to format (can be None)
            precision: Number of decimal places
            suffix: Optional suffix to add (e.g., 's' for seconds)
            
        Returns:
            Formatted string or "N/A" if value is None
        """
        if value is None:
            return "N/A"
        try:
            float_val = float(value)
            # Check for NaN or infinity
            if not (float_val == float_val):  # NaN check
                return "N/A"
            if float_val in [float('inf'), float('-inf')]:
                return "N/A"
            return f"{float_val:.{precision}f}{suffix}"
        except (ValueError, TypeError):
            return "N/A"

    def load_data(self, source: str = "database") -> pd.DataFrame:
        """
        Load experiment data from specified source

        Args:
            source: "database" or "csv"

        Returns:
            DataFrame containing experiment data
        """
        try:
            if source == "database" and Path(self.db_path).exists():
                conn = sqlite3.connect(self.db_path)
                self.df = pd.read_sql_query("SELECT * FROM experiments", conn)
                conn.close()
                st.success(f"✅ Loaded {len(self.df)} experiments from database")

            elif source == "csv" and Path(self.csv_path).exists():
                self.df = pd.read_csv(self.csv_path)
                st.success(f"✅ Loaded {len(self.df)} experiments from CSV")

            else:
                st.error(f"❌ Data source '{source}' not found!")
                return pd.DataFrame()

            # Data preprocessing
            if not self.df.empty:
                self._preprocess_data()

            return self.df
        
        except Exception as e:
            st.error(f"❌ Error loading data: {e}")
            return pd.DataFrame()

    def _preprocess_data(self):
        """Preprocess the loaded data"""
        # Remove duplicate rows by row_id if it exists, otherwise by all columns
        if 'row_id' in self.df.columns:
            initial_count = len(self.df)
            self.df = self.df.drop_duplicates(subset=['row_id'], keep='last')
            if len(self.df) < initial_count:
                st.info(f"ℹ️ Removed {initial_count - len(self.df)} duplicate experiments (by row_id)")
        else:
            # If no row_id, remove completely identical rows
            initial_count = len(self.df)
            self.df = self.df.drop_duplicates(keep='last')
            if len(self.df) < initial_count:
                st.info(f"ℹ️ Removed {initial_count - len(self.df)} duplicate rows")
        
        # Convert timestamp to datetime
        if 'timestamp' in self.df.columns:
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], errors='coerce')

        # Parse JSON columns
        json_columns = ['model_hyperparameters', 'all_test_metrics', 'experiment_files']
        for col in json_columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].apply(self._safe_json_loads)

        # Fill NaN values
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        self.df[numeric_columns] = self.df[numeric_columns].fillna(0)
        

    def _safe_json_loads(self, json_str):
        """Safely parse JSON string"""
        try:
            return json.loads(json_str) if pd.notna(json_str) else {}
        except:
            return {}

    def show_experiment_details(self, experiment_row):
        """Show detailed view of a selected experiment with improved layout"""
        # Add back button at the top of the page
        if st.button("← Back to Dashboard", type="secondary", key="top_back_button"):
            st.session_state.show_experiment_details = False
            st.session_state.selected_experiment = None
            st.rerun()
        
        st.header(f"🔍 Experiment Details: {experiment_row['model'].upper()} - {experiment_row['run_id']}")
        
        # Basic information in organized cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Model", experiment_row['model'].upper())
            st.metric("Dataset", experiment_row.get('dataset', 'N/A'))
            st.metric("Classification Type", "Multilabel" if experiment_row.get('is_multilabel', False) else "Binary")
        
        with col2:
            st.metric("Accuracy", self.safe_format_metric(experiment_row.get('accuracy')))
            st.metric("F1 Score", self.safe_format_metric(experiment_row.get('macro_f1')))
            st.metric("Precision", self.safe_format_metric(experiment_row.get('macro_precision')))
        
        with col3:
            st.metric("Recall", self.safe_format_metric(experiment_row.get('macro_recall')))
            st.metric("Training Time", self.safe_format_metric(experiment_row.get('total_training_time'), precision=2, suffix="s"))
            st.metric("Inference Time", self.safe_format_metric(experiment_row.get('total_inference_time'), precision=2, suffix="s"))

        # Add Linear Probe Metrics section
        if any(experiment_row.get(f'linear_probe_{metric}') is not None for metric in ['accuracy', 'f1', 'precision', 'recall', 'auc']):
            st.markdown("---")
            st.subheader("🔬 Linear Probe Metrics")
            
            with st.expander("ℹ️ What are Linear Probe Metrics?", expanded=False):
                st.markdown("""
                **Linear probing** evaluates the quality of learned representations by training a simple linear classifier 
                on top of frozen model embeddings. These metrics indicate how well the model has learned meaningful features:
                
                - **LP Accuracy**: Classification accuracy using only a linear layer
                - **LP F1/Precision/Recall**: Standard classification metrics for the linear probe
                - **LP AUC**: Area under the ROC curve for the linear probe  
                - **LP CV Accuracy**: Cross-validation accuracy (mean ± std) for robust evaluation
                
                Higher linear probe metrics suggest better representation learning.
                """)
            
            lp_col1, lp_col2, lp_col3 = st.columns(3)
            
            with lp_col1:
                st.metric("LP Accuracy", self.safe_format_metric(experiment_row.get('linear_probe_accuracy')))
                st.metric("LP F1 Score", self.safe_format_metric(experiment_row.get('linear_probe_f1')))
            
            with lp_col2:
                st.metric("LP Precision", self.safe_format_metric(experiment_row.get('linear_probe_precision')))
                st.metric("LP Recall", self.safe_format_metric(experiment_row.get('linear_probe_recall')))
            
            with lp_col3:
                st.metric("LP AUC", self.safe_format_metric(experiment_row.get('linear_probe_auc')))
                cv_accuracy = experiment_row.get('linear_probe_cv_accuracy')
                cv_std = experiment_row.get('linear_probe_cv_std')
                if cv_accuracy is not None and cv_std is not None:
                    cv_display = f"{cv_accuracy:.3f} ± {cv_std:.3f}"
                else:
                    cv_display = self.safe_format_metric(cv_accuracy)
                st.metric("LP CV Accuracy", cv_display)

        # Add spacing between sections
        st.markdown("---")

        # Display plots if available
        self._display_experiment_plots(experiment_row)
        
        # Display confusion matrices
        self._display_confusion_matrices(experiment_row)
        
        # Display classification report
        self._display_classification_report(experiment_row)

    def _display_experiment_plots(self, experiment_row):
        """Display training plots with consistent sizing and layout"""
        st.subheader("📈 Training Plots")
        
        experiment_files = experiment_row.get('experiment_files', {})
        if not experiment_files:
            st.info("No plot files found for this experiment")
            return
        
        # Find different types of plot files
        training_plots = {k: v for k, v in experiment_files.items() if k.startswith('plot_') and 'tsne' not in k}
        tsne_main_plot = {k: v for k, v in experiment_files.items() if k.startswith('plot_tsne_')}
        tsne_class_plots = {k: v for k, v in experiment_files.items() if k.startswith('tsne_class_')}
        
        # Display training plots (loss curves, accuracy, etc.)
        if training_plots:
            st.subheader("📊 Training Performance")
            cols = st.columns(min(len(training_plots), 3))
            
            for idx, (plot_key, plot_path) in enumerate(training_plots.items()):
                col_idx = idx % 3
                
                try:
                    absolute_path = Path(plot_path)
                    
                    if absolute_path.exists():
                        with cols[col_idx]:
                            st.subheader(plot_key.replace('plot_', '').replace('_', ' ').title())
                            image = Image.open(absolute_path)
                            # Set consistent width for training plots
                            st.image(image, width=400)
                    else:
                        with cols[col_idx]:
                            st.warning(f"Plot file not found: {absolute_path}")
                except Exception as e:
                    with cols[col_idx]:
                        st.error(f"Error loading plot {plot_key}: {e}")
            
            # Add spacing after training plots
            st.markdown("---")
        
        # Display t-SNE visualizations
        if tsne_main_plot or tsne_class_plots:
            st.subheader("🎯 t-SNE Embeddings Visualization")
            
            # Display main t-SNE plot first
            if tsne_main_plot:
                plot_key, plot_path = next(iter(tsne_main_plot.items()))
                try:
                    absolute_path = Path(plot_path)
                    if absolute_path.exists():
                        st.subheader("📈 t-SNE Overview")
                        image = Image.open(absolute_path)
                        # Set consistent width for t-SNE overview - limit very large plots
                        st.image(image, width=800, caption="Overall t-SNE visualization showing embedding clusters")
                    else:
                        st.warning(f"Main t-SNE plot not found: {absolute_path}")
                except Exception as e:
                    st.error(f"Error loading main t-SNE plot: {e}")
                
                # Add spacing after main t-SNE plot
                if tsne_class_plots:
                    st.markdown("") 
            
            # Display per-class t-SNE plots
            if tsne_class_plots:
                st.subheader("🔍 Per-Class t-SNE Analysis")
                st.info("Individual t-SNE plots for each diagnostic class showing positive vs negative samples")
                
                # Sort class plots by name for consistent display
                sorted_class_plots = dict(sorted(tsne_class_plots.items()))
                
                # Display in rows of 2 columns for better layout
                plot_items = list(sorted_class_plots.items())
                for i in range(0, len(plot_items), 2):
                    row_plots = plot_items[i:i+2]
                    cols = st.columns(len(row_plots))
                    
                    for col_idx, (plot_key, plot_path) in enumerate(row_plots):
                        try:
                            absolute_path = Path(plot_path)
                            
                            if absolute_path.exists():
                                with cols[col_idx]:
                                    # Extract class name from key (e.g., 'tsne_class_cd_tsne' -> 'CD')
                                    class_name = plot_key.replace('tsne_class_', '').replace('_tsne', '').upper()
                                    st.subheader(f"📊 {class_name}")
                                    image = Image.open(absolute_path)
                                    # Set consistent width for per-class t-SNE plots
                                    st.image(image, width=500, caption=f"{class_name} class separation")
                            else:
                                with cols[col_idx]:
                                    st.warning(f"Class plot not found: {absolute_path}")
                        except Exception as e:
                            with cols[col_idx]:
                                st.error(f"Error loading class plot {plot_key}: {e}")
            
            # Add spacing after t-SNE section
            st.markdown("---")
        
        # Show message if no plots found
        if not training_plots and not tsne_main_plot and not tsne_class_plots:
            st.info("No training plots found")

    def _display_confusion_matrices(self, experiment_row):
        """Display confusion matrices with consistent sizing"""
        st.subheader("🔲 Confusion Matrices")
        
        experiment_files = experiment_row.get('experiment_files', {})
        confusion_files = {k: v for k, v in experiment_files.items() if 'confusion' in k}
        
        if not confusion_files:
            st.info("No confusion matrix files found")
            return
        
        # Display confusion matrices
        for conf_key, conf_path in confusion_files.items():
            try:
                # Use the absolute path directly
                absolute_path = Path(conf_path)
                
                if absolute_path.exists():
                    st.subheader(conf_key.replace('confusion_', '').replace('_', ' ').title())
                    image = Image.open(absolute_path)
                    # Set consistent width for confusion matrices based on aspect ratio
                    img_width, img_height = image.size
                    aspect_ratio = img_width / img_height
                    
                    # Adjust width based on aspect ratio to maintain readability
                    if aspect_ratio > 1.5:  # Wide confusion matrix (e.g., multilabel)
                        display_width = 900
                    else:  # Square or portrait confusion matrix
                        display_width = 600
                    
                    st.image(image, width=display_width)
                else:
                    st.warning(f"Confusion matrix file not found: {absolute_path}")
            except Exception as e:
                st.error(f"Error loading confusion matrix {conf_key}: {e}")
        
        # Add spacing after confusion matrices
        st.markdown("---")

    def _display_classification_report(self, experiment_row):
        """Display classification report from metrics"""
        st.subheader("📊 Classification Report")
        
        all_metrics = experiment_row.get('all_test_metrics', {})
        
        if not all_metrics or 'classification_report' not in all_metrics:
            st.warning("No classification report found in metrics")
            return
        
        try:
            classification_report = all_metrics['classification_report']
            
            # Convert to DataFrame for better display
            report_data = []
            for class_name, metrics in classification_report.items():
                if isinstance(metrics, dict):
                    row = {'Class': class_name}
                    row.update(metrics)
                    report_data.append(row)
            
            if report_data:
                report_df = pd.DataFrame(report_data)
                
                # Format numeric columns
                numeric_cols = ['precision', 'recall', 'f1-score', 'support']
                for col in numeric_cols:
                    if col in report_df.columns:
                        report_df[col] = pd.to_numeric(report_df[col], errors='coerce')
                        if col != 'support':
                            report_df[col] = report_df[col].round(4)
                
                st.dataframe(report_df, width='stretch', hide_index=True)
            else:
                st.warning("Could not parse classification report")
                
        except Exception as e:
            st.error(f"Error displaying classification report: {e}")
            
        # Also display raw metrics for reference
        with st.expander("🔍 Raw Metrics Data"):
            st.json(all_metrics)

    def show_overview(self):
        """Show overview statistics"""
        st.header("📊 Experiment Overview")

        if self.df is None or self.df.empty:
            st.warning("No data available")
            return

        # Check if is_multilabel column exists
        if 'is_multilabel' not in self.df.columns:
            st.warning("No 'is_multilabel' column found in data")
            return

        # Create two-column layout for binary and multilabel classification
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🔵 Binary Classification")
            # Filter for binary classification experiments (is_multilabel == False)
            binary_df = self.df[self.df['is_multilabel'] == False]
            
            if binary_df.empty:
                st.warning("No binary classification experiments found")
            else:
                self._show_filtered_models_performance(binary_df, "Binary Classification")

        with col2:
            st.subheader("🟡 Multilabel Classification")
            # Filter for multilabel classification experiments (is_multilabel == True)
            multilabel_df = self.df[self.df['is_multilabel'] == True]
            
            if multilabel_df.empty:
                st.warning("No multilabel classification experiments found")
            else:
                self._show_filtered_models_performance(multilabel_df, "Multilabel Classification")

    def _show_filtered_models_performance(self, filtered_df, classification_type):
        """Show performance for models in filtered dataframe"""
        if 'model' not in filtered_df.columns:
            st.warning("No model data available")
            return
            
        # Get all unique models in filtered data
        models = filtered_df['model'].unique()
        
        # Create performance summary table - ensure unique rows by row_id
        performance_data = []
        seen_row_ids = set()  # Track seen row_ids to avoid duplicates
        
        for model in models:
            model_df = filtered_df[filtered_df['model'] == model]
            if model_df.empty:
                continue
            
            # Find the best performing experiment for this model (prioritize accuracy, then F1)
            best_experiment = None
            
            # First try to find best by accuracy
            if 'accuracy' in filtered_df.columns and not model_df['accuracy'].isna().all():
                best_experiment = model_df.loc[model_df['accuracy'].idxmax()]
            # If no accuracy data, try F1 score
            elif 'macro_f1' in filtered_df.columns and not model_df['macro_f1'].isna().all():
                best_experiment = model_df.loc[model_df['macro_f1'].idxmax()]
            
            if best_experiment is not None:
                # Check if we've already processed this row_id
                row_id = best_experiment.get('row_id', f"{model}_{best_experiment.name}")
                if row_id in seen_row_ids:
                    continue
                seen_row_ids.add(row_id)
                
                # Create performance data row
                perf_data = {'Model': model.upper()}
                
                # Add only the most important metrics for overview
                for metric_col, display_name in [
                    ('accuracy', 'Accuracy'),
                    ('macro_f1', 'F1 Score'),
                    ('linear_probe_accuracy', 'LP Accuracy'),
                    ('linear_probe_f1', 'LP F1')
                ]:
                    if metric_col in filtered_df.columns and pd.notna(best_experiment[metric_col]):
                        perf_data[display_name] = f"{best_experiment[metric_col]:.4f}"
                    else:
                        perf_data[display_name] = "N/A"
                
                # Add run information
                if 'run_id' in filtered_df.columns and pd.notna(best_experiment.get('run_id')):
                    perf_data['Run ID'] = best_experiment['run_id']
                else:
                    perf_data['Run ID'] = "N/A"
                
                performance_data.append(perf_data)
        
        # Display performance table
        if performance_data:
            performance_df = pd.DataFrame(performance_data)
            st.dataframe(performance_df, width='stretch')
            
            # Show best performing models for this classification type
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🥇 Best Accuracy")
                if 'accuracy' in filtered_df.columns and not filtered_df['accuracy'].isna().all():
                    best_acc = filtered_df.loc[filtered_df['accuracy'].idxmax()]
                    st.metric(
                        label=f"{best_acc['model'].upper()}",
                        value=f"{best_acc['accuracy']:.4f}",
                        help=f"Best Accuracy for {classification_type}"
                    )
                else:
                    st.metric("N/A", "No accuracy data")
            
            with col2:
                st.subheader("🥇 Best F1 Score")
                if 'macro_f1' in filtered_df.columns:
                    valid_f1 = filtered_df.dropna(subset=['macro_f1'])
                    if not valid_f1.empty:
                        best_f1 = valid_f1.loc[valid_f1['macro_f1'].idxmax()]
                        st.metric(
                            label=f"{best_f1['model'].upper()}",
                            value=f"{best_f1['macro_f1']:.4f}",
                            help=f"Best F1 Score for {classification_type}"
                        )
                    else:
                        st.metric("N/A", "No F1 data")
                else:
                    st.metric("N/A", "No F1 data")



    def show_detailed_view(self):
        """Show detailed view of all experiments"""
        st.header("🔍 Detailed Experiment View")

        if self.df is None or self.df.empty:
            st.warning("No data available")
            return

        # Filters
        st.subheader("Filters")
        col1, col2, col3 = st.columns(3)

        with col1:
            if 'model' in self.df.columns:
                models = ['All'] + list(self.df['model'].unique())
                selected_model = st.selectbox("Model", models)

        with col2:
            if 'dataset' in self.df.columns:
                datasets = ['All'] + list(self.df['dataset'].unique())
                selected_dataset = st.selectbox("Dataset", datasets)

        with col3:
            search_term = st.text_input("Search (run_id, etc.)")

        # Apply filters
        filtered_df = self.df.copy()

        if 'model' in self.df.columns and selected_model != 'All':
            filtered_df = filtered_df[filtered_df['model'] == selected_model]

        if 'dataset' in self.df.columns and selected_dataset != 'All':
            filtered_df = filtered_df[filtered_df['dataset'] == selected_dataset]

        if search_term:
            mask = filtered_df.astype(str).apply(
                lambda x: x.str.contains(search_term, case=False, na=False)
            ).any(axis=1)
            filtered_df = filtered_df[mask]

        # Sort options
        sort_col1, sort_col2 = st.columns(2)
        with sort_col1:
            sort_options = ['timestamp', 'accuracy', 'linear_probe_accuracy', 'macro_f1', 'linear_probe_f1', 'total_parameters', 'run_id']
            available_sort = [col for col in sort_options if col in filtered_df.columns]
            if available_sort:
                sort_by = st.selectbox("Sort by", available_sort)

        with sort_col2:
            sort_order = st.selectbox("Order", ["Descending", "Ascending"])

        # Sort dataframe
        if available_sort:
            ascending = sort_order == "Ascending"
            filtered_df = filtered_df.sort_values(sort_by, ascending=ascending)

        # Display results
        st.subheader(f"Experiments ({len(filtered_df)} results)")
        
        # Add info about linear probe metrics
        if any(col.startswith('linear_probe') for col in filtered_df.columns):
            with st.expander("ℹ️ About Linear Probe (LP) Metrics", expanded=False):
                st.markdown("""
                **Linear Probe metrics** evaluate representation quality by training a simple linear classifier 
                on frozen model embeddings. LP metrics prefixed with "LP" show how well the model learned features 
                that can be linearly separated. Higher LP metrics indicate better representation learning.
                """)

        # Column selection with performance-focused defaults
        if not filtered_df.empty:
            all_columns = list(filtered_df.columns)
            default_columns = [
                'timestamp', 'run_id', 'model', 'is_multilabel', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'macro_auc',
                'linear_probe_accuracy', 'linear_probe_f1', 'linear_probe_precision', 'linear_probe_recall', 'linear_probe_auc', 'linear_probe_cv_accuracy',
                'total_training_time', 'total_inference_time'
            ]
            available_default = [col for col in default_columns if col in all_columns]

            selected_columns = st.multiselect(
                "Select columns to display",
                all_columns,
                default=available_default
            )

            if selected_columns:
                display_df = filtered_df[selected_columns].copy()
            else:
                display_df = filtered_df.copy()

            # Prepare column configuration for better display
            column_config = {}
            if 'is_multilabel' in display_df.columns:
                column_config['is_multilabel'] = st.column_config.CheckboxColumn(
                    "Multilabel",
                    help="Whether experiment uses multilabel classification",
                    disabled=True
                )
            
            # Add linear probe metric column configurations
            linear_probe_columns = {
                'linear_probe_accuracy': st.column_config.NumberColumn(
                    "LP Accuracy",
                    help="Linear Probe Accuracy",
                    format="%.4f"
                ),
                'linear_probe_f1': st.column_config.NumberColumn(
                    "LP F1",
                    help="Linear Probe F1 Score",
                    format="%.4f"
                ),
                'linear_probe_precision': st.column_config.NumberColumn(
                    "LP Precision",
                    help="Linear Probe Precision",
                    format="%.4f"
                ),
                'linear_probe_recall': st.column_config.NumberColumn(
                    "LP Recall",
                    help="Linear Probe Recall",
                    format="%.4f"
                ),
                'linear_probe_auc': st.column_config.NumberColumn(
                    "LP AUC",
                    help="Linear Probe AUC",
                    format="%.4f"
                ),
                'linear_probe_cv_accuracy': st.column_config.NumberColumn(
                    "LP CV Acc",
                    help="Linear Probe Cross-Validation Accuracy",
                    format="%.4f"
                )
            }
            
            for col, config in linear_probe_columns.items():
                if col in display_df.columns:
                    column_config[col] = config
            
            # Use st.data_editor with selection to make rows clickable
            edited_df = st.data_editor(
                display_df, 
                width='stretch', 
                height=600,
                column_config=column_config,
                hide_index=True,
                disabled=True,  # Make read-only
                key="experiment_table"
            )

            # Download option
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Filtered Data as CSV",
                data=csv,
                file_name=f"filtered_experiments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )


def main():
    """Main dashboard application"""
    st.title("🔬 ML Experiment Dashboard")
    st.markdown("Track and analyze your machine learning experiments")

    # Initialize session state
    if 'show_experiment_details' not in st.session_state:
        st.session_state.show_experiment_details = False
    if 'selected_experiment' not in st.session_state:
        st.session_state.selected_experiment = None

    # Initialize dashboard
    dashboard = ExperimentDashboard()

    # Sidebar - always show
    st.sidebar.header("📊 Data Source")

    # Data source selection
    data_source = st.sidebar.radio(
        "Choose data source",
        ["Database", "CSV", "Refresh Data"]
    )

    # Load data based on selection
    if data_source == "Refresh Data":
        st.sidebar.info("Refreshing data from experiments folder...")
        try:
            from scripts.experiment_data_parser import ExperimentDataParser
            parser = ExperimentDataParser()
            df = parser.update_experiment_data()
            dashboard.df = df
            st.sidebar.success("✅ Data refreshed successfully!")
        except ImportError:
            st.sidebar.error("❌ experiment_data_parser module not found")
        except Exception as e:
            st.sidebar.error(f"❌ Error refreshing data: {e}")

    elif data_source == "Database":
        dashboard.load_data("database")

    elif data_source == "CSV":
        dashboard.load_data("csv")

    # Experiment Selection Section in Sidebar
    if dashboard.df is not None and not dashboard.df.empty:
        st.sidebar.header("🔍 View Experiment Details")
        
        # Model selection dropdown in sidebar
        available_models = list(dashboard.df['model'].unique())
        selected_model_for_details = st.sidebar.selectbox(
            "Select Model:",
            available_models,
            key="sidebar_model_selection"
        )
        
        # Run number input in sidebar
        run_number_input = st.sidebar.text_input(
            "Enter run number:",
            placeholder="e.g., 1, 2, 3",
            key="sidebar_run_number_input"
        )
        
        # Find matching experiments based on both model and run number
        selected_experiment = None
        
        if run_number_input:
            # Filter by the selected model
            search_df = dashboard.df[dashboard.df['model'] == selected_model_for_details]
            
            # Look for experiments that match the run number
            matching_experiments = search_df[
                search_df['run_id'].str.contains(f"run_{run_number_input}$", case=False, na=False)
            ]
            
            if len(matching_experiments) > 0:
                if len(matching_experiments) == 1:
                    selected_experiment = matching_experiments.iloc[0]
                    st.sidebar.success(f"✅ Found: {selected_experiment['model'].upper()} - run_{run_number_input}")
                else:
                    st.sidebar.info(f"📋 Found {len(matching_experiments)} experiments")
                    for idx, exp in matching_experiments.iterrows():
                        st.sidebar.write(f"• {exp['model'].upper()} - {exp['run_id']}")
                    selected_experiment = matching_experiments.iloc[0]  # Select first match
                    st.sidebar.info(f"🎯 Auto-selected: {selected_experiment['model'].upper()}")
            else:
                st.sidebar.warning(f"❌ No {selected_model_for_details} run_{run_number_input} found")
        
        # View details button in sidebar
        if selected_experiment is not None:
            if st.sidebar.button("View Details", type="primary", width='stretch', key="sidebar_view_details"):
                # Store selected experiment in session state and switch to details view
                st.session_state.selected_experiment = selected_experiment.to_dict()
                st.session_state.show_experiment_details = True
                st.rerun()
        else:
            st.sidebar.button("View Details", type="primary", width='stretch', disabled=True, key="sidebar_view_details_disabled")

    # Main content area
    if st.session_state.show_experiment_details and st.session_state.selected_experiment:
        # Show experiment details with sidebar still visible
        dashboard.show_experiment_details(st.session_state.selected_experiment)
    else:
        # Show main dashboard content
        if dashboard.df is not None and not dashboard.df.empty:
            tab1, tab2 = st.tabs(["📊 Overview", "🔍 Detailed View"])

            with tab1:
                dashboard.show_overview()

            with tab2:
                dashboard.show_detailed_view()
        else:
            st.warning("No experiment data available. Please check your data source.")
            st.info("**Getting Started:**")
            st.markdown("""
            1. Run some experiments using `experiment_runner.py`
            2. Parse the data using `experiment_data_parser.py` 
            3. Or click 'Refresh Data' in the sidebar to automatically parse
            """)


if __name__ == "__main__":
    main()
