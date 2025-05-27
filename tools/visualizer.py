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
    merged_paths = defaultdict(list)
    
    # First, group tensors by normalized paths
    for tensor in tensors:
        module_path = tensor.get('module', 'Unknown')
        if not module_path or module_path == 'Unknown':
            continue
        
        # Normalize the path for merging similar modules
        normalized_path = normalize_module_path(module_path)
        merged_paths[normalized_path].append(tensor)
    
    # Now build the tree using the normalized paths
    for normalized_path, path_tensors in merged_paths.items():
        # Get the original first path to extract components
        # We'll use the normalized path for display but components for hierarchy
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

def create_module_tree_chart(device_data: Dict) -> None:
    """Create and display a tree chart for module hierarchy."""
    tensors = device_data.get('tensors', [])
    module_tree = build_module_tree(tensors)
    
    st.header("Module Hierarchy")

    # Flatten the tree for display
    flattened_nodes = []
    
    def flatten_tree(tree, prefix="", depth=0):
        for key, value in sorted(tree.items(), key=lambda x: x[0]):
            if key == '_tensors' or key == '_size' or key == '_pattern':
                continue
                
            size = format_bytes(value.get('_size', 0))
            full_path = f"{prefix}.{key}" if prefix else key
            
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
                "display_path": display_path
            })
            
            # Recursively process children
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
        # A node represents a merged pattern if its display_path (derived from pattern) contains '*'
        # and it's not a leaf tensor group itself.
        # The node['display_path'] is the key, as it's set to the pattern path if merged.
        if '*' in node['display_path']:
            merged_module_data.append({
                'Pattern': node['display_path'], 
                'Size (Bytes)': node['size_bytes'],
                'Size (Formatted)': node['size_formatted']
            })

    if merged_module_data:
        merged_df = pd.DataFrame(merged_module_data)
        # Sort by size for better visualization
        merged_df = merged_df.sort_values('Size (Bytes)', ascending=False)

        st.subheader("Memory Usage by Merged Module Patterns")
        fig = px.bar(
            merged_df,
            x='Pattern',
            y='Size (Bytes)',
            title='Memory Usage by Merged Module Patterns',
            labels={'Size (Bytes)': 'Memory Usage (bytes)', 'Pattern': 'Module Pattern'},
            hover_data=['Size (Formatted)']
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Display the merged module data in a table
        st.subheader("Merged Module Patterns - Table View")
        st.dataframe(
            merged_df[['Pattern', 'Size (Formatted)', 'Size (Bytes)']].rename(
                columns={'Pattern': 'Module Pattern', 'Size (Formatted)': 'Memory Usage'}
            ),
            hide_index=True, 
            use_container_width=True,
            column_config={
                "Size (Bytes)": None # Hide the raw bytes column from display but keep for sorting
            }
        )
    # --- End new code ---
    
    # Display each module as a separate expander (not nested)
    for node in flattened_nodes:
        expander = st.expander(f"{node['name']} (Path: {node['display_path']})")
        with expander:
            if node["tensors"]:
                # Check for merged tensors
                merged_info = ""
                sample_tensor = node["tensors"][0] if node["tensors"] else None
                if sample_tensor and sample_tensor.get('_merged', False):
                    merged_count = sample_tensor.get('_merged_count', 0)
                    merged_pattern = sample_tensor.get('_merged_pattern', '')
                    merged_info = f"⚠️ This represents {merged_count} similar modules matching pattern: {merged_pattern}"
                    st.info(merged_info)
                
                t_df = pd.DataFrame([{
                    'Size': format_bytes(t.get('size_bytes', 0)),
                    'Shape': t.get('shape', 'Unknown'),
                    'Module': t.get('module', 'Unknown')  # Show full module path
                } for t in node["tensors"]])
                
                if not t_df.empty:
                    st.dataframe(t_df)

def create_phase_breakdown(tensors: List[Dict]) -> None:
    """Create and display a breakdown of tensors by phase."""
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
        
        # Create bar chart
        fig = px.bar(
            phase_df, 
            x='Phase', 
            y='Total Size (bytes)',
            title='Memory Usage by Phase',
            labels={'Total Size (bytes)': 'Memory Usage (bytes)'},
            hover_data=['Tensor Count', 'Total Size']
        )
        st.plotly_chart(fig)
        
        # Show detailed table
        st.dataframe(phase_df[['Phase', 'Total Size', 'Tensor Count']], hide_index=True)

def display_tensor_details(tensors: List[Dict], min_size_mb: float = 0) -> None:
    """Display detailed information about tensors."""
    min_bytes = min_size_mb * 1024 * 1024
    filtered_tensors = [t for t in tensors if t.get('size_bytes', 0) >= min_bytes]
    
    if not filtered_tensors:
        st.warning(f"No tensors found with size >= {min_size_mb} MB")
        return
    
    tensor_data = []
    for tensor in filtered_tensors:
        tensor_data.append({
            'Size (MB)': tensor.get('size_mb', 0),
            'Shape': tensor.get('shape', 'Unknown'),
            'Module': tensor.get('module', 'Unknown'),
            'Phase': tensor.get('phase', 'Unknown'),
            'Creation Stack': '\n'.join(tensor.get('create_stack_trace', [])[:3]),  # First 3 lines
            'Release Stack': '\n'.join(tensor.get('release_stack_trace', [])[:3])   # First 3 lines
        })
    
    tensor_df = pd.DataFrame(tensor_data)
    tensor_df = tensor_df.sort_values('Size (MB)', ascending=False)
    
    st.dataframe(tensor_df, use_container_width=True)

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
        
        # Add a filter for minimum tensor size
        min_size_mb = st.slider(
            "Minimum Tensor Size (MB)", 
            0.0, 
            max(1.0, peak_memory_mb), 
            1.0
        )
        
        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs(["Phases", "Module Tree", "Stack Trace Groups", "Tensor Details"])
        
        with tab1:
            create_phase_breakdown(tensors)
        
        with tab2:
            create_module_tree_chart(device_data)
        
        with tab3:
            display_stack_trace_groups(tensors)
        
        with tab4:
            display_tensor_details(tensors, min_size_mb)

if __name__ == "__main__":
    main()
