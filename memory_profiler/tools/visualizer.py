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

def load_memory_snapshot(file_path: str) -> Dict:
    """Load memory snapshot from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def extract_module_hierarchy(module_path: str) -> List[str]:
    """Extract module hierarchy components from a module path."""
    return module_path.split('.')

def build_module_tree(tensors: List[Dict]) -> Dict:
    """Build a hierarchical tree of modules from tensor data."""
    tree = {}
    for tensor in tensors:
        module_path = tensor.get('module', 'Unknown')
        if not module_path or module_path == 'Unknown':
            continue
            
        components = extract_module_hierarchy(module_path)
        current = tree
        for component in components:
            if component not in current:
                current[component] = {'_tensors': [], '_size': 0}
            current = current[component]
        
        current['_tensors'].append(tensor)
        current['_size'] += tensor.get('size_bytes', 0)
    
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
            if key == '_tensors' or key == '_size':
                continue
                
            size = format_bytes(value.get('_size', 0))
            node_name = f"{'  ' * depth}📂 {key} - {size}"
            
            # Store node info
            flattened_nodes.append({
                "name": node_name,
                "depth": depth,
                "size_bytes": value.get('_size', 0),
                "size_formatted": size,
                "tensors": value.get('_tensors', []),
                "full_path": f"{prefix}.{key}" if prefix else key
            })
            
            # Recursively process children
            new_prefix = f"{prefix}.{key}" if prefix else key
            flatten_tree(value, new_prefix, depth + 1)
    
    # Generate the flattened representation
    flatten_tree(module_tree)
    
    # Display each module as a separate expander (not nested)
    for node in flattened_nodes:
        expander = st.expander(node["name"])
        with expander:
            if node["tensors"]:
                t_df = pd.DataFrame([{
                    'Size': format_bytes(t.get('size_bytes', 0)),
                    'Shape': t.get('shape', 'Unknown'),
                    'Module': t.get('module', 'Unknown').split('.')[-1]
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

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="PyTorch Memory Profiler Visualizer")
    parser.add_argument("--file", "-f", type=str, help="Path to memory snapshot JSON file")
    
    # Extract args from sys.argv, excluding Streamlit's own arguments
    streamlit_args = []
    file_arg = None
    
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] in ["--file", "-f"] and i + 1 < len(sys.argv):
            file_arg = sys.argv[i + 1]
            i += 2
        else:
            streamlit_args.append(sys.argv[i])
            i += 1
    
    # Override sys.argv for Streamlit
    sys.argv = [sys.argv[0]] + streamlit_args
    
    st.set_page_config(
        page_title="PyTorch Memory Visualizer",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    st.title("PyTorch Memory Profiler Visualizer")
    
    # Load snapshot from CLI argument if provided
    snapshot_data = None
    if file_arg and os.path.exists(file_arg):
        try:
            with open(file_arg, 'r') as f:
                snapshot_data = json.load(f)
            st.success(f"Loaded memory snapshot from: {file_arg}")
        except Exception as e:
            st.error(f"Failed to load file: {e}")
    
    # Fallback to file uploader if no valid file from CLI
    if snapshot_data is None:
        uploaded_file = st.file_uploader("Upload a memory snapshot JSON file", type=['json'])
        if uploaded_file is not None:
            # Load the snapshot data
            snapshot_data = json.load(uploaded_file)
    
    # Process the snapshot data if available
    if snapshot_data is not None:
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
