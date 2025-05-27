import streamlit as st
import json
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import defaultdict
import re
import sys
import argparse
from typing import Dict, List, Any, Tuple, Set, Optional

# Disable Streamlit file watcher to avoid "inotify watch limit reached" error
os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'

def load_memory_snapshot(file_path: str) -> Dict:
    """Load memory snapshot from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def extract_module_hierarchy(module_path: str) -> List[str]:
    """Extract module hierarchy components from a module path."""
    return module_path.split('.')

def normalize_module_path(module_path: str) -> str:
    """Normalize a module path by replacing numeric indices with placeholders."""
    # Split the path by dots
    components = module_path.split('.')
    
    # Replace numeric components with '*'
    normalized_components = []
    for component in components:
        if component.isdigit():
            normalized_components.append('*')
        else:
            normalized_components.append(component)
    
    # Join the components back with dots
    normalized_path = '.'.join(normalized_components)
    
    return normalized_path

def build_module_tree(tensors: List[Dict]) -> Dict:
    """Build a hierarchical tree of modules from tensor data."""
    tree = {}
    
    # Special grouping for specific phases
    special_groups = {
        'setup_model_and_optimizer': [],
        'optimizer_step_iter_0': [],
        'forward_backward_unknown': []  # For Unknown modules in forward_backward_iter_0 and forward_backward_iter_1
    }
    
    # Regular tensors for normal tree building
    regular_tensors = []
    
    # First pass: separate special groups from regular tensors
    for tensor in tensors:
        phase = tensor.get('phase', '')
        module_path = tensor.get('module', 'Unknown')
        
        # Check for special phase groupings
        if phase == 'setup_model_and_optimizer':
            special_groups['setup_model_and_optimizer'].append(tensor)
        elif phase == 'optimizer_step_iter_0':
            special_groups['optimizer_step_iter_0'].append(tensor)
        elif (phase in ['forward_backward_iter_0', 'forward_backward_iter_1'] and 
              (module_path == 'Unknown' or not module_path)):
            special_groups['forward_backward_unknown'].append(tensor)
        else:
            # Regular tensor, process normally
            if module_path and module_path != 'Unknown':
                regular_tensors.append(tensor)
    
    # Add special groups to the tree at the root level
    for group_name, group_tensors in special_groups.items():
        if group_tensors:
            total_size = sum(t.get('size_bytes', 0) for t in group_tensors)
            tree[group_name] = {
                '_tensors': group_tensors,
                '_size': total_size,
                '_pattern': group_name,
                '_special_group': True  # Mark as special group
            }
    
    # Process regular tensors with the original logic
    merged_paths = defaultdict(list)
    
    # Group regular tensors by normalized paths
    for tensor in regular_tensors:
        module_path = tensor.get('module', 'Unknown')
        # Normalize the path for merging similar modules
        normalized_path = normalize_module_path(module_path)
        merged_paths[normalized_path].append(tensor)
    
    # Now build the tree using the normalized paths
    for normalized_path, path_tensors in merged_paths.items():
        # Get the original first path to extract components
        sample_path = path_tensors[0].get('module', 'Unknown')
        components = extract_module_hierarchy(sample_path)
        
        # Calculate total size of all tensors in this merged group
        total_size = sum(tensor.get('size_bytes', 0) for tensor in path_tensors)
        tensor_count = len(path_tensors)
        
        # Identify the pattern in the original path
        pattern = sample_path
        if len(path_tensors) > 1:  # If we have multiple tensors, it's a merged group
            # Use the normalized path directly as it now uses '*' for numeric components
            pattern = normalized_path
        
        # Add a custom property to the first tensor to mark it as merged with count info
        if len(path_tensors) > 1:
            path_tensors[0]['_merged'] = True
            path_tensors[0]['_merged_count'] = tensor_count
            path_tensors[0]['_merged_pattern'] = pattern
        
        # Now build tree structure
        current = tree
        for i, component in enumerate(components):
            # Replace numeric component with a pattern indicator in the tree
            is_numeric = component.isdigit()
            tree_key = component
            
            # Create or get the node
            if tree_key not in current:
                current[tree_key] = {'_tensors': [], '_size': 0, '_pattern': None}
            
            current = current[tree_key]

        # Store tensors and size in the leaf node
        current['_tensors'].extend(path_tensors)
        current['_size'] = total_size
        current['_pattern'] = pattern
    
    return tree

def group_by_stack_trace(tensors: List[Dict]) -> Dict[str, List[Dict]]:
    """Group tensors by their creation stack trace."""
    grouped = defaultdict(list)

    for tensor in tensors:
        # Create a hash from the stack trace (joining all lines)
        stack_trace = tensor.get('create_stack_trace', [])
        stack_key = ''.join(stack_trace)

        grouped[stack_key].append(tensor)

    return grouped

def group_by_phase(tensors: List[Dict]) -> Dict[str, List[Dict]]:
    """Group tensors by their creation phase."""
    grouped = defaultdict(list)
    
    for tensor in tensors:
        phase = tensor.get('phase', 'Unknown')
        grouped[phase].append(tensor)
    
    return grouped

def format_bytes(size_bytes: float) -> str:
    """Format bytes into human-readable format."""
    if size_bytes < 1024:
        return f"{size_bytes:.2f} B"
    elif size_bytes < 1024**2:
        return f"{size_bytes/1024:.2f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes/1024**2:.2f} MB"
    else:
        return f"{size_bytes/1024**3:.2f} GB"

def create_module_tree_chart(device_data: Dict, tensors: List[Dict]) -> None:
    """Create and display a tree chart for module hierarchy with phase breakdown."""
    module_tree = build_module_tree(tensors)
    
    # First, show the phase breakdown
    st.header("Memory Analysis")
    
    # Create phase breakdown chart
    phase_groups = group_by_phase(tensors)
    
    # Create DataFrame for phase breakdown
    phase_sizes = []
    for phase, phase_tensors in phase_groups.items():
        total_size = sum(t.get('size_bytes', 0) for t in phase_tensors)
        num_tensors = len(phase_tensors)
        phase_sizes.append({
            'Phase': phase,
            'Total Size (bytes)': total_size,
            'Total Size': format_bytes(total_size),
            'Tensor Count': num_tensors
        })
    
    phase_df = pd.DataFrame(phase_sizes)
    if not phase_df.empty:
        # Sort by size
        phase_df = phase_df.sort_values('Total Size (bytes)', ascending=False)
        
        # Create bar chart with text annotations
        fig = px.bar(
            phase_df, 
            x='Phase', 
            y='Total Size (bytes)',
            title='Memory Usage by Phase',
            labels={'Total Size (bytes)': 'Memory Usage (bytes)'},
            hover_data=['Tensor Count', 'Total Size'],
            text='Total Size'  # Display formatted size on bars
        )
        
        # Update text position and formatting
        fig.update_traces(
            texttemplate='%{text}',
            textposition='outside',
            textfont_size=12
        )
        
        # Update layout for better visibility
        fig.update_layout(
            showlegend=False,
            yaxis_title="Memory Usage (bytes)",
            xaxis_title="Phase",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Now show the module hierarchy
    st.header("Module Hierarchy")

    # Flatten the tree for display
    flattened_nodes = []
    
    def flatten_tree(tree, prefix="", depth=0):
        for key, value in sorted(tree.items(), key=lambda x: x[0]):
            if key == '_tensors' or key == '_size' or key == '_pattern' or key == '_special_group':
                continue
                
            size = format_bytes(value.get('_size', 0))
            full_path = f"{prefix}.{key}" if prefix else key
            
            # Check if this is a special group
            is_special_group = value.get('_special_group', False)
            
            # Format the node name based on type
            if is_special_group:
                # Special formatting for phase-based groups
                if key == 'setup_model_and_optimizer':
                    display_name = "Setup Model & Optimizer (All Tensors)"
                elif key == 'optimizer_step_iter_0':
                    display_name = "Optimizer Step Iter 0 (All Tensors)"
                elif key == 'forward_backward_unknown':
                    display_name = "Forward/Backward Unknown Tensors"
                else:
                    display_name = key
                
                node_name = f"{'  ' * depth}🔥 {display_name} - {size}"
                display_path = key
            else:
                # Check if this is a merged pattern node
                pattern = value.get('_pattern')
                if pattern and '*' in pattern:
                    node_name = f"{'  ' * depth}📂 {key} - {size} [Merged Pattern]"
                    display_path = pattern
                else:
                    node_name = f"{'  ' * depth}📂 {key} - {size}"
                    display_path = full_path
            
            # Store node info
            flattened_nodes.append({
                "name": node_name,
                "depth": depth,
                "size_bytes": value.get('_size', 0),
                "size_formatted": size,
                "tensors": value.get('_tensors', []),
                "full_path": full_path,
                "display_path": display_path,
                "is_special_group": is_special_group
            })
            
            # Recursively process children (only if not a special group)
            if not is_special_group:
                new_prefix = full_path
                flatten_tree(value, new_prefix, depth + 1)
    
    # Generate the flattened representation
    flatten_tree(module_tree)

    # Calculate total memory from flattened_nodes
    total_module_tree_memory = sum(node['size_bytes'] for node in flattened_nodes)
    st.metric(label="Total Memory in Module Tree", value=format_bytes(total_module_tree_memory))
    
    # --- New code for merged modules chart ---
    merged_module_data = []
    for node in flattened_nodes:
        # Include both merged patterns and special groups
        if '*' in node['display_path'] or node.get('is_special_group', False):
            # Format the pattern name for special groups
            if node.get('is_special_group', False):
                if node['display_path'] == 'setup_model_and_optimizer':
                    pattern_name = "Setup Model & Optimizer (All Tensors)"
                elif node['display_path'] == 'optimizer_step_iter_0':
                    pattern_name = "Optimizer Step Iter 0 (All Tensors)"
                elif node['display_path'] == 'forward_backward_unknown':
                    pattern_name = "Forward/Backward Unknown Tensors"
                else:
                    pattern_name = node['display_path']
            else:
                pattern_name = node['display_path']
            
            merged_module_data.append({
                'Pattern': pattern_name, 
                'Size (Bytes)': node['size_bytes'],
                'Size (Formatted)': node['size_formatted'],
                'Is Special Group': node.get('is_special_group', False)
            })

    if merged_module_data:
        merged_df = pd.DataFrame(merged_module_data)
        # Sort by size for better visualization
        merged_df = merged_df.sort_values('Size (Bytes)', ascending=False)

        st.subheader("Memory Usage by Module Groups")
        
        # Create color mapping for special groups vs regular patterns
        color_map = {True: '#FF6B6B', False: '#4ECDC4'}  # Red for special groups, teal for patterns
        merged_df['Color'] = merged_df['Is Special Group'].map(color_map)
        
        fig = px.bar(
            merged_df,
            x='Pattern',
            y='Size (Bytes)',
            title='Memory Usage by Module Groups and Patterns',
            labels={'Size (Bytes)': 'Memory Usage (bytes)', 'Pattern': 'Module/Group'},
            hover_data=['Size (Formatted)'],
            color='Is Special Group',
            color_discrete_map={True: '#FF6B6B', False: '#4ECDC4'}
        )
        
        # Update layout for better readability
        fig.update_layout(
            xaxis_tickangle=-45,
            height=600,
            showlegend=True,
            legend_title_text='Type',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99
            )
        )
        
        # Update legend labels
        fig.for_each_trace(lambda t: t.update(name='Special Group' if t.name == 'True' else 'Merged Pattern'))
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display the merged module data in a table with selection
        st.subheader("Module Groups and Patterns - Table View")
        
        # Add select all checkbox
        col1, col2 = st.columns([1, 5])
        with col1:
            select_all = st.checkbox("Select All", key="select_all_checkbox")
        
        # Create a data editor with selection enabled
        display_df = merged_df[['Pattern', 'Size (Formatted)', 'Size (Bytes)', 'Is Special Group']].copy()
        
        # Add type indicator to the display
        display_df['Type'] = display_df['Is Special Group'].map({True: '🔥 Special Group', False: '📂 Pattern'})
        
        # Select columns for display
        display_df = display_df[['Pattern', 'Type', 'Size (Formatted)', 'Size (Bytes)']].rename(
            columns={'Pattern': 'Module/Group', 'Size (Formatted)': 'Memory Usage'}
        ).reset_index(drop=True)
        
        # Add a selection column - set initial value based on select_all
        display_df.insert(0, 'Select', select_all)
        
        # Initialize session state for tracking manual edits
        if 'prev_select_all' not in st.session_state:
            st.session_state.prev_select_all = False
        
        # Check if select_all state changed
        if select_all != st.session_state.prev_select_all:
            st.session_state.prev_select_all = select_all
            # Force refresh by clearing any cached state
            if 'merged_patterns_table' in st.session_state:
                del st.session_state['merged_patterns_table']
        
        edited_df = st.data_editor(
            display_df,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Select": st.column_config.CheckboxColumn(
                    "Select",
                    help="Select rows to calculate total memory",
                    default=select_all,
                ),
                "Size (Bytes)": None,  # Hide the raw bytes column from display but keep for calculation
                "Module/Group": st.column_config.TextColumn(
                    "Module/Group",
                    help="Module pattern or special group",
                    disabled=True,
                ),
                "Type": st.column_config.TextColumn(
                    "Type",
                    help="Type of grouping",
                    disabled=True,
                ),
                "Memory Usage": st.column_config.TextColumn(
                    "Memory Usage", 
                    help="Formatted memory usage",
                    disabled=True,
                )
            },
            key="merged_patterns_table"
        )
        
        # Calculate and display total memory for selected rows
        selected_rows = edited_df[edited_df['Select'] == True]
        if not selected_rows.empty:
            total_selected_bytes = selected_rows['Size (Bytes)'].sum()
            total_selected_formatted = format_bytes(total_selected_bytes)
            
            # Display the total in a prominent way
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                # Check if all items are selected
                all_selected = len(selected_rows) == len(display_df)
                if all_selected:
                    label_text = f"Total Memory for All {len(selected_rows)} Items"
                else:
                    label_text = f"Total Memory for {len(selected_rows)} Selected Items"
                
                st.metric(
                    label=label_text,
                    value=total_selected_formatted,
                    help=f"Sum of {len(selected_rows)} selected items"
                )
            
            # Show details of selected patterns
            with st.expander(f"Details of {len(selected_rows)} Selected Items"):
                selected_details = selected_rows[['Module/Group', 'Type', 'Memory Usage']].copy()
                st.dataframe(selected_details, hide_index=True, use_container_width=True)
        else:
            st.info("💡 Select one or more patterns above to see the total memory usage")
    # --- End new code ---
    
    # Display each module as a separate expander (not nested)
    for node in flattened_nodes:
        expander = st.expander(f"{node['name']} (Path: {node['display_path']})")
        with expander:
            if node["tensors"]:
                # Special handling for phase-grouped tensors
                if node.get('is_special_group', False):
                    st.info(f"📊 This group contains {len(node['tensors'])} tensors from the {node['display_path']} phase")
                    
                    # Group tensors by module for better display
                    module_groups = defaultdict(list)
                    for t in node["tensors"]:
                        module = t.get('module', 'Unknown')
                        module_groups[module].append(t)
                    
                    # Display summary by module
                    st.write(f"**Modules in this group:** {len(module_groups)}")
                    
                    # Create detailed dataframe with all tensor information
                    tensor_details = []
                    for t in node["tensors"]:
                        # Process creation stack trace to show relevant portion
                        stack_trace = t.get('create_stack_trace', [])
                        processed_stack = []
                        
                        # Find the last line containing megatron/core/*.py
                        last_megatron_idx = -1
                        for i in range(len(stack_trace) - 1, -1, -1):
                            if 'megatron/core/' in stack_trace[i] and '.py' in stack_trace[i]:
                                last_megatron_idx = i
                                break
                        
                        # If found, get 10 lines from that point upward
                        if last_megatron_idx >= 0:
                            start_idx = max(0, last_megatron_idx - 9)
                            processed_stack = stack_trace[start_idx:last_megatron_idx + 1]
                        else:
                            # If no megatron/core line found, show last 10 lines
                            processed_stack = stack_trace[-10:] if len(stack_trace) > 10 else stack_trace
                        
                        tensor_details.append({
                            'Size (MB)': t.get('size_mb', 0),
                            'Size': format_bytes(t.get('size_bytes', 0)),
                            'Shape': t.get('shape', 'Unknown'),
                            'Module': t.get('module', 'Unknown'),
                            'Phase': t.get('phase', 'Unknown'),
                            'Creation Stack': '\n'.join(processed_stack)
                        })
                    
                    t_df = pd.DataFrame(tensor_details)
                    # Sort by size
                    if 'Size (MB)' in t_df.columns:
                        t_df = t_df.sort_values('Size (MB)', ascending=False)
                else:
                    # Check for merged tensors
                    merged_info = ""
                    sample_tensor = node["tensors"][0] if node["tensors"] else None
                    if sample_tensor and sample_tensor.get('_merged', False):
                        merged_count = sample_tensor.get('_merged_count', 0)
                        merged_pattern = sample_tensor.get('_merged_pattern', '')
                        merged_info = f"⚠️ This represents {merged_count} similar modules matching pattern: {merged_pattern}"
                        st.info(merged_info)
                    
                    # Create detailed dataframe with all tensor information
                    tensor_details = []
                    for t in node["tensors"]:
                        # Process creation stack trace
                        stack_trace = t.get('create_stack_trace', [])
                        processed_stack = []
                        
                        # Find the last line containing megatron/core/*.py
                        last_megatron_idx = -1
                        for i in range(len(stack_trace) - 1, -1, -1):
                            if 'megatron/core/' in stack_trace[i] and '.py' in stack_trace[i]:
                                last_megatron_idx = i
                                break
                        
                        # If found, get 10 lines from that point upward
                        if last_megatron_idx >= 0:
                            start_idx = max(0, last_megatron_idx - 9)
                            processed_stack = stack_trace[start_idx:last_megatron_idx + 1]
                        else:
                            # If no megatron/core line found, show last 10 lines
                            processed_stack = stack_trace[-10:] if len(stack_trace) > 10 else stack_trace
                        
                        tensor_details.append({
                            'Size (MB)': t.get('size_mb', 0),
                            'Size': format_bytes(t.get('size_bytes', 0)),
                            'Shape': t.get('shape', 'Unknown'),
                            'Module': t.get('module', 'Unknown'),
                            'Phase': t.get('phase', 'Unknown'),
                            'Creation Stack': '\n'.join(processed_stack)
                        })
                    
                    t_df = pd.DataFrame(tensor_details)
                    # Sort by size
                    if 'Size (MB)' in t_df.columns:
                        t_df = t_df.sort_values('Size (MB)', ascending=False)
                
                if not t_df.empty:
                    # Configure column display
                    column_config = {
                        "Size (MB)": st.column_config.NumberColumn(
                            "Size (MB)",
                            help="Tensor size in megabytes",
                            format="%.2f"
                        ),
                        "Creation Stack": st.column_config.TextColumn(
                            "Creation Stack",
                            help="Stack trace from tensor creation (last 10 lines from megatron/core)",
                            width="large"
                        )
                    }
                    
                    # Display dataframe with expandable rows for stack traces
                    st.dataframe(
                        t_df,
                        column_config=column_config,
                        use_container_width=True,
                        hide_index=True
                    )





def display_stack_trace_groups(tensors: List[Dict]) -> None:
    """Display tensors grouped by stack traces."""
    stack_groups = group_by_stack_trace(tensors)
    
    # Sort groups by total size
    sorted_groups = sorted(
        stack_groups.items(), 
        key=lambda x: sum(t.get('size_bytes', 0) for t in x[1]),
        reverse=True
    )
    
    st.header("Tensors Grouped by Creation Stack Trace")
    
    for i, (stack_key, group_tensors) in enumerate(sorted_groups[:20]):  # Limit to top 20
        total_size = sum(t.get('size_bytes', 0) for t in group_tensors)
        expander = st.expander(
            f"Group {i+1}: {len(group_tensors)} tensors, Total: {format_bytes(total_size)}"
        )
        
        with expander:
            # Display the stack trace
            if group_tensors and 'create_stack_trace' in group_tensors[0]:
                st.code('\n'.join(group_tensors[0]['create_stack_trace'][:5]))
            
            # Display tensor details
            tensor_data = []
            for tensor in group_tensors:
                tensor_data.append({
                    'Size': format_bytes(tensor.get('size_bytes', 0)),
                    'Shape': tensor.get('shape', 'Unknown'),
                    'Module': tensor.get('module', 'Unknown').split('.')[-1],
                    'Phase': tensor.get('phase', 'Unknown')
                })
            
            tensor_df = pd.DataFrame(tensor_data)
            st.dataframe(tensor_df, hide_index=True)

# Process command-line arguments before Streamlit runs
@st.cache_data
def parse_command_line_args():
    """Parse command-line arguments without modifying sys.argv"""
    # Look for our custom arguments without disturbing sys.argv
    file_arg = None
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] in ["--file", "-f"] and i + 1 < len(sys.argv):
            file_arg = sys.argv[i + 1]
            break
        i += 1
    
    return file_arg

# Load JSON snapshot file
def load_snapshot_from_file(file_path):
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Failed to load file: {e}")
    return None

def main():
    st.set_page_config(
        page_title="PyTorch Memory Visualizer",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    st.title("PyTorch Memory Profiler Visualizer")
    
    # Get command-line argument
    file_arg = parse_command_line_args()
    
    # Initialize session state for snapshot data if not present
    if 'snapshot_data' not in st.session_state:
        st.session_state.snapshot_data = None
    
    # Try to load from command-line argument first
    if file_arg and (st.session_state.snapshot_data is None):
        snapshot_data = load_snapshot_from_file(file_arg)
        if snapshot_data:
            st.success(f"Loaded memory snapshot from: {file_arg}")
            st.session_state.snapshot_data = snapshot_data
    
    # Handle file uploader
    uploaded_file = st.file_uploader("Upload a memory snapshot JSON file", type=['json'])
    if uploaded_file is not None:
        try:
            # Load the snapshot data from uploaded file
            snapshot_data = json.load(uploaded_file)
            st.session_state.snapshot_data = snapshot_data
        except Exception as e:
            st.error(f"Error loading uploaded file: {e}")
    
    # Process snapshot data if available
    if st.session_state.snapshot_data:
        snapshot_data = st.session_state.snapshot_data
        
        # Show device selection if multiple devices
        devices = list(snapshot_data.keys())
        
        if len(devices) > 1:
            selected_device = st.selectbox("Select Device", devices)
        else:
            selected_device = devices[0]
        
        device_data = snapshot_data[selected_device]
        tensors = device_data.get('tensors', [])
        peak_memory_mb = device_data.get('peak_memory_mb', 0)
        
        st.header(f"Peak Memory: {peak_memory_mb:.2f} MB on {selected_device}")
        
        # Display the module tree view with integrated phase breakdown
        create_module_tree_chart(device_data, tensors)

if __name__ == "__main__":
    main()
