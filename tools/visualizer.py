"""
PyTorch Memory Profiler Visualizer

A Streamlit-based tool for visualizing PyTorch memory snapshots with module hierarchy,
phase breakdown, and tensor grouping capabilities.
"""

import streamlit as st
import json
import os
import pandas as pd
import plotly.express as px
from collections import defaultdict
import sys
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

# Disable Streamlit file watcher to avoid "inotify watch limit reached" error
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"


# ============================================================================
# Constants and Configuration
# ============================================================================


class SpecialPhase(Enum):
    """Special phases that get grouped separately in the module tree."""

    SETUP_MODEL_OPTIMIZER = "setup_model_and_optimizer"
    OPTIMIZER_STEP_ITER_0 = "optimizer_step_iter_0"
    FORWARD_BACKWARD_UNKNOWN = "forward_backward_unknown"


SPECIAL_GROUP_DISPLAY_NAMES = {
    SpecialPhase.SETUP_MODEL_OPTIMIZER.value: "Setup Model & Optimizer (All Tensors)",
    SpecialPhase.OPTIMIZER_STEP_ITER_0.value: "Optimizer Step Iter 0 (All Tensors)",
    SpecialPhase.FORWARD_BACKWARD_UNKNOWN.value: "Forward/Backward Unknown Tensors",
}

# UI Configuration
PHASE_CHART_HEIGHT = 500
MODULE_CHART_HEIGHT = 600
MAX_STACK_TRACE_GROUPS = 20
STACK_TRACE_PREVIEW_LINES = 5
MEGATRON_STACK_TRACE_LINES = 10

# Colors
COLOR_SPECIAL_GROUP = "#FF6B6B"  # Red
COLOR_MERGED_PATTERN = "#4ECDC4"  # Teal


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class TensorInfo:
    """Represents information about a single tensor."""

    size_bytes: int
    size_mb: float
    shape: str
    module: str
    phase: str
    create_stack_trace: List[str]

    @classmethod
    def from_dict(cls, data: Dict) -> "TensorInfo":
        """Create TensorInfo from dictionary."""
        return cls(
            size_bytes=data.get("size_bytes", 0),
            size_mb=data.get("size_mb", 0),
            shape=data.get("shape", "Unknown"),
            module=data.get("module", "Unknown"),
            phase=data.get("phase", "Unknown"),
            create_stack_trace=data.get("create_stack_trace", []),
        )


@dataclass
class ModuleNode:
    """Represents a node in the module hierarchy tree."""

    tensors: List[Dict]
    size: int
    pattern: Optional[str]
    is_special_group: bool = False

    def to_dict(self) -> Dict:
        """Convert to dictionary for tree representation."""
        return {
            "_tensors": self.tensors,
            "_size": self.size,
            "_pattern": self.pattern,
            "_special_group": self.is_special_group,
        }


# ============================================================================
# File I/O Functions
# ============================================================================


def load_memory_snapshot(file_path: str) -> Dict:
    """Load memory snapshot from JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)


def load_snapshot_from_file(file_path: str) -> Optional[Dict]:
    """
    Load snapshot from file with error handling.

    Args:
        file_path: Path to the JSON snapshot file

    Returns:
        Loaded snapshot data or None if error
    """
    if os.path.exists(file_path):
        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Failed to load file: {e}")
    return None


# ============================================================================
# Module Path Processing Functions
# ============================================================================


def extract_module_hierarchy(module_path: str) -> List[str]:
    """Extract module hierarchy components from a module path."""
    return module_path.split(".")


def normalize_module_path(module_path: str) -> str:
    """
    Normalize a module path by replacing numeric indices with placeholders.

    Example:
        'model.layers.0.attention' -> 'model.layers.*.attention'
    """
    components = module_path.split(".")
    normalized_components = [
        "*" if component.isdigit() else component for component in components
    ]
    return ".".join(normalized_components)


# ============================================================================
# Tensor Grouping Functions
# ============================================================================


def group_by_phase(tensors: List[Dict]) -> Dict[str, List[Dict]]:
    """Group tensors by their creation phase."""
    grouped = defaultdict(list)
    for tensor in tensors:
        phase = tensor.get("phase", "Unknown")
        grouped[phase].append(tensor)
    return grouped


def group_by_stack_trace(tensors: List[Dict]) -> Dict[str, List[Dict]]:
    """Group tensors by their creation stack trace."""
    grouped = defaultdict(list)
    for tensor in tensors:
        stack_trace = tensor.get("create_stack_trace", [])
        stack_key = "".join(stack_trace)
        grouped[stack_key].append(tensor)
    return grouped


def classify_tensor_by_phase(tensor: Dict) -> Optional[str]:
    """
    Classify a tensor into special phase groups.

    Returns:
        Special group name if tensor belongs to one, None otherwise
    """
    phase = tensor.get("phase", "")
    module_path = tensor.get("module", "Unknown")

    if phase == SpecialPhase.SETUP_MODEL_OPTIMIZER.value:
        return SpecialPhase.SETUP_MODEL_OPTIMIZER.value
    elif phase == SpecialPhase.OPTIMIZER_STEP_ITER_0.value:
        return SpecialPhase.OPTIMIZER_STEP_ITER_0.value
    elif phase in ["forward_backward_iter_0", "forward_backward_iter_1"] and (
        module_path == "Unknown" or not module_path
    ):
        return SpecialPhase.FORWARD_BACKWARD_UNKNOWN.value

    return None


# ============================================================================
# Module Tree Building Functions
# ============================================================================


def separate_special_and_regular_tensors(
    tensors: List[Dict],
) -> Tuple[Dict[str, List[Dict]], List[Dict]]:
    """
    Separate tensors into special groups and regular tensors.

    Returns:
        Tuple of (special_groups_dict, regular_tensors_list)
    """
    special_groups = {
        SpecialPhase.SETUP_MODEL_OPTIMIZER.value: [],
        SpecialPhase.OPTIMIZER_STEP_ITER_0.value: [],
        SpecialPhase.FORWARD_BACKWARD_UNKNOWN.value: [],
    }
    regular_tensors = []

    for tensor in tensors:
        special_group = classify_tensor_by_phase(tensor)
        if special_group:
            special_groups[special_group].append(tensor)
        else:
            module_path = tensor.get("module", "Unknown")
            if module_path and module_path != "Unknown":
                regular_tensors.append(tensor)

    return special_groups, regular_tensors


def add_special_groups_to_tree(
    tree: Dict, special_groups: Dict[str, List[Dict]]
) -> None:
    """Add special phase groups to the module tree at root level."""
    for group_name, group_tensors in special_groups.items():
        if group_tensors:
            total_size = sum(t.get("size_bytes", 0) for t in group_tensors)
            node = ModuleNode(
                tensors=group_tensors,
                size=total_size,
                pattern=group_name,
                is_special_group=True,
            )
            tree[group_name] = node.to_dict()


def merge_similar_modules(tensors: List[Dict]) -> Dict[str, List[Dict]]:
    """Group tensors by normalized module paths for merging similar modules."""
    merged_paths = defaultdict(list)

    for tensor in tensors:
        module_path = tensor.get("module", "Unknown")
        normalized_path = normalize_module_path(module_path)
        merged_paths[normalized_path].append(tensor)

    return merged_paths


def mark_merged_tensors(path_tensors: List[Dict], pattern: str) -> None:
    """Mark the first tensor in a group as merged with count info."""
    if len(path_tensors) > 1:
        path_tensors[0]["_merged"] = True
        path_tensors[0]["_merged_count"] = len(path_tensors)
        path_tensors[0]["_merged_pattern"] = pattern


def build_tree_path(
    tree: Dict,
    components: List[str],
    tensors: List[Dict],
    total_size: int,
    pattern: str,
) -> None:
    """Build a path in the tree for the given components."""
    current = tree

    for component in components:
        if component not in current:
            current[component] = {"_tensors": [], "_size": 0, "_pattern": None}
        current = current[component]

    # Store tensors and metadata in the leaf node
    current["_tensors"].extend(tensors)
    current["_size"] = total_size
    current["_pattern"] = pattern


def build_module_tree(tensors: List[Dict]) -> Dict:
    """
    Build a hierarchical tree of modules from tensor data.

    This function creates a tree structure where:
    - Special phase groups are at the root level
    - Similar modules (differing only by numeric indices) are merged
    - Each node contains tensors, total size, and pattern information
    """
    tree = {}

    # Separate special groups from regular tensors
    special_groups, regular_tensors = separate_special_and_regular_tensors(tensors)

    # Add special groups to tree
    add_special_groups_to_tree(tree, special_groups)

    # Process regular tensors
    merged_paths = merge_similar_modules(regular_tensors)

    # Build tree from merged paths
    for normalized_path, path_tensors in merged_paths.items():
        sample_path = path_tensors[0].get("module", "Unknown")
        components = extract_module_hierarchy(sample_path)

        total_size = sum(tensor.get("size_bytes", 0) for tensor in path_tensors)

        # Determine pattern
        pattern = normalized_path if len(path_tensors) > 1 else sample_path

        # Mark merged tensors
        mark_merged_tensors(path_tensors, pattern)

        # Build tree structure
        build_tree_path(tree, components, path_tensors, total_size, pattern)

    return tree


# ============================================================================
# Formatting Functions
# ============================================================================


def format_bytes(size_bytes: float) -> str:
    """Format bytes into human-readable format."""
    units = [(1024**3, "GB"), (1024**2, "MB"), (1024, "KB"), (1, "B")]

    for unit_size, unit_name in units:
        if size_bytes >= unit_size:
            return f"{size_bytes/unit_size:.2f} {unit_name}"

    return f"{size_bytes:.2f} B"


# ============================================================================
# Stack Trace Processing Functions
# ============================================================================


def process_stack_trace_for_display(
    stack_trace: List[str],
    target_pattern: str = "megatron/core/",
    num_lines: int = MEGATRON_STACK_TRACE_LINES,
) -> List[str]:
    """
    Process stack trace to show relevant portion.

    Args:
        stack_trace: Full stack trace lines
        target_pattern: Pattern to search for in stack trace
        num_lines: Number of lines to show

    Returns:
        Processed stack trace lines
    """
    # Find the last line containing the target pattern
    last_target_idx = -1
    for i in range(len(stack_trace) - 1, -1, -1):
        if target_pattern in stack_trace[i] and ".py" in stack_trace[i]:
            last_target_idx = i
            break

    # Extract relevant lines
    if last_target_idx >= 0:
        start_idx = max(0, last_target_idx - num_lines + 1)
        return stack_trace[start_idx : last_target_idx + 1]
    else:
        # If pattern not found, show last n lines
        return stack_trace[-num_lines:] if len(stack_trace) > num_lines else stack_trace


# ============================================================================
# UI Component Functions
# ============================================================================


def create_phase_breakdown_chart(phase_groups: Dict[str, List[Dict]]) -> None:
    """Create and display a bar chart showing memory usage by phase."""
    # Prepare data
    phase_sizes = []
    for phase, phase_tensors in phase_groups.items():
        total_size = sum(t.get("size_bytes", 0) for t in phase_tensors)
        phase_sizes.append(
            {
                "Phase": phase,
                "Total Size (bytes)": total_size,
                "Total Size": format_bytes(total_size),
                "Tensor Count": len(phase_tensors),
            }
        )

    phase_df = pd.DataFrame(phase_sizes)
    if phase_df.empty:
        return

    # Sort by size
    phase_df = phase_df.sort_values("Total Size (bytes)", ascending=False)

    # Truncate long phase names
    def truncate_phase(phase, max_length=30):
        if len(phase) <= max_length:
            return phase
        return phase[:max_length-3] + "..."
    
    phase_df["Display Phase"] = phase_df["Phase"].apply(lambda x: truncate_phase(x))

    # Create chart
    fig = px.bar(
        phase_df,
        x="Display Phase",
        y="Total Size (bytes)",
        title="Memory Usage by Phase",
        labels={"Total Size (bytes)": "Memory Usage", "Display Phase": "Phase"},
        hover_data={"Phase": True, "Tensor Count": True, "Total Size": True, "Display Phase": False},
        text="Total Size",
        color_discrete_sequence=["#4ECDC4"]  # Use consistent color scheme
    )

    # Update styling to match the module chart
    fig.update_traces(
        texttemplate='%{text}', 
        textposition='outside', 
        textfont_size=10,
        marker_line_color='rgba(0,0,0,0.2)',
        marker_line_width=1,
        hovertemplate='<b>%{customdata[0]}</b><br>' +
                      'Memory: %{customdata[2]}<br>' +
                      'Tensor Count: %{customdata[1]}<br>' +
                      '<extra></extra>'
    )

    fig.update_layout(
        showlegend=False,
        yaxis_title="Memory Usage",
        xaxis_title="Phase",
        height=PHASE_CHART_HEIGHT,
        xaxis_tickangle=-45,
        margin=dict(l=80, r=80, t=100, b=120),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12),
        title=dict(
            font=dict(size=16),
            x=0.5,
            xanchor='center'
        ),
        yaxis=dict(
            gridcolor='rgba(128,128,128,0.2)',
            tickformat='.2s',  # Use SI prefix notation
            title_font=dict(size=14)
        ),
        xaxis=dict(
            title_font=dict(size=14),
            tickfont=dict(size=11)
        )
    )

    st.plotly_chart(fig, use_container_width=True)


def flatten_module_tree(tree: Dict) -> List[Dict]:
    """
    Flatten the module tree into a list of nodes for display.

    Returns:
        List of dictionaries containing node information
    """
    flattened_nodes = []

    def flatten_tree_recursive(subtree: Dict, prefix: str = "", depth: int = 0) -> None:
        for key, value in sorted(subtree.items(), key=lambda x: x[0]):
            # Skip internal keys
            if key.startswith("_"):
                continue

            size = format_bytes(value.get("_size", 0))
            full_path = f"{prefix}.{key}" if prefix else key
            is_special_group = value.get("_special_group", False)

            # Format display name
            if is_special_group:
                display_name = SPECIAL_GROUP_DISPLAY_NAMES.get(key, key)
                node_name = f"{'  ' * depth}🔥 {display_name} - {size}"
                display_path = key
            else:
                pattern = value.get("_pattern")
                if pattern and "*" in pattern:
                    node_name = f"{'  ' * depth}📂 {key} - {size} [Merged Pattern]"
                    display_path = pattern
                else:
                    node_name = f"{'  ' * depth}📂 {key} - {size}"
                    display_path = full_path

            # Add node to flattened list
            flattened_nodes.append(
                {
                    "name": node_name,
                    "depth": depth,
                    "size_bytes": value.get("_size", 0),
                    "size_formatted": size,
                    "tensors": value.get("_tensors", []),
                    "full_path": full_path,
                    "display_path": display_path,
                    "is_special_group": is_special_group,
                }
            )

            # Recurse for non-special groups
            if not is_special_group:
                flatten_tree_recursive(value, full_path, depth + 1)

    flatten_tree_recursive(tree)
    return flattened_nodes


def create_merged_modules_chart(flattened_nodes: List[Dict]) -> None:
    """Create and display chart for merged module patterns."""
    # Extract merged module data
    merged_module_data = []
    for node in flattened_nodes:
        if "*" in node["display_path"] or node.get("is_special_group", False):
            pattern_name = (
                SPECIAL_GROUP_DISPLAY_NAMES.get(
                    node["display_path"], node["display_path"]
                )
                if node.get("is_special_group", False)
                else node["display_path"]
            )

            merged_module_data.append(
                {
                    "Pattern": pattern_name,
                    "Size (Bytes)": node["size_bytes"],
                    "Size (Formatted)": node["size_formatted"],
                    "Is Special Group": node.get("is_special_group", False),
                }
            )

    if not merged_module_data:
        return

    # Create DataFrame and sort
    merged_df = pd.DataFrame(merged_module_data)
    merged_df = merged_df.sort_values("Size (Bytes)", ascending=False)

    st.subheader("Memory Usage by Module Groups")

    # Truncate long pattern names for display
    def truncate_pattern(pattern, max_length=40):
        if len(pattern) <= max_length:
            return pattern
        # Try to truncate at a meaningful boundary
        parts = pattern.split('.')
        truncated = pattern[:max_length-3] + "..."
        # Try to find a better truncation point at a dot
        for i in range(max_length-3, 0, -1):
            if pattern[i] == '.':
                truncated = pattern[:i] + "..."
                break
        return truncated
    
    # Create a copy for display with truncated names
    display_df = merged_df.copy()
    display_df["Display Pattern"] = display_df["Pattern"].apply(lambda x: truncate_pattern(x))
    
    # Create bar chart
    fig = px.bar(
        display_df,
        x="Display Pattern",
        y="Size (Bytes)",
        title="Memory Usage by Module Groups and Patterns",
        labels={"Size (Bytes)": "Memory Usage", "Display Pattern": "Module/Group"},
        hover_data={"Pattern": True, "Size (Formatted)": True, "Display Pattern": False},
        color="Is Special Group",
        color_discrete_map={True: COLOR_SPECIAL_GROUP, False: COLOR_MERGED_PATTERN},
        text="Size (Formatted)"
    )

    # Update layout for better display
    fig.update_layout(
        xaxis_tickangle=-45,
        height=MODULE_CHART_HEIGHT,
        showlegend=True,
        legend_title_text="Type",
        legend=dict(
            yanchor="top", 
            y=0.99, 
            xanchor="right", 
            x=0.99,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="rgba(0, 0, 0, 0.2)",
            borderwidth=1
        ),
        margin=dict(l=80, r=80, t=100, b=120),  # Increase bottom margin for labels
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12),
        title=dict(
            font=dict(size=16),
            x=0.5,
            xanchor='center'
        ),
        yaxis=dict(
            gridcolor='rgba(128,128,128,0.2)',
            tickformat='.2s',  # Use SI prefix notation (K, M, G)
            title_font=dict(size=14)
        ),
        xaxis=dict(
            title_font=dict(size=14),
            tickfont=dict(size=11)
        )
    )

    # Update bar appearance
    fig.update_traces(
        texttemplate='%{text}',
        textposition='outside',
        textfont_size=10,
        marker_line_color='rgba(0,0,0,0.2)',
        marker_line_width=1
    )

    # Update legend labels
    fig.for_each_trace(
        lambda t: t.update(
            name="Special Group" if t.name == "True" else "Merged Pattern"
        )
    )

    # Add hover template for better tooltip
    fig.update_traces(
        hovertemplate='<b>%{customdata[0]}</b><br>' +
                      'Memory: %{customdata[1]}<br>' +
                      '<extra></extra>'
    )

    st.plotly_chart(fig, use_container_width=True)

    # Display table with selection
    display_merged_modules_table(merged_df)


def display_merged_modules_table(merged_df: pd.DataFrame) -> None:
    """Display an interactive table for merged modules with selection capability."""
    st.subheader("Module Groups and Patterns - Table View")

    # Add select all checkbox
    col1, col2 = st.columns([1, 5])
    with col1:
        select_all = st.checkbox("Select All", key="select_all_checkbox")

    # Prepare display DataFrame
    display_df = merged_df[
        ["Pattern", "Size (Formatted)", "Size (Bytes)", "Is Special Group"]
    ].copy()
    display_df["Type"] = display_df["Is Special Group"].map(
        {True: "🔥 Special Group", False: "📂 Pattern"}
    )

    display_df = (
        display_df[["Pattern", "Type", "Size (Formatted)", "Size (Bytes)"]]
        .rename(columns={"Pattern": "Module/Group", "Size (Formatted)": "Memory Usage"})
        .reset_index(drop=True)
    )

    display_df.insert(0, "Select", select_all)

    # Handle select all state changes
    if "prev_select_all" not in st.session_state:
        st.session_state.prev_select_all = False

    if select_all != st.session_state.prev_select_all:
        st.session_state.prev_select_all = select_all
        if "merged_patterns_table" in st.session_state:
            del st.session_state["merged_patterns_table"]

    # Display editable table with height set to show all rows
    # Calculate height based on number of rows (approximately 35px per row + header)
    table_height = max(400, min(1000, 35 * (len(display_df) + 1)))
    
    edited_df = st.data_editor(
        display_df,
        hide_index=True,
        use_container_width=True,
        height=table_height,  # Set height to show all rows
        column_config={
            "Select": st.column_config.CheckboxColumn(
                "Select",
                help="Select rows to calculate total memory",
                default=select_all,
            ),
            "Size (Bytes)": None,  # Hide raw bytes column
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
            ),
        },
        key="merged_patterns_table",
    )

    # Display selection summary
    display_selection_summary(edited_df, display_df)


def display_selection_summary(
    edited_df: pd.DataFrame, display_df: pd.DataFrame
) -> None:
    """Display summary of selected items."""
    selected_rows = edited_df[edited_df["Select"] == True]

    if selected_rows.empty:
        st.info("💡 Select one or more patterns above to see the total memory usage")
        return

    total_selected_bytes = selected_rows["Size (Bytes)"].sum()
    total_selected_formatted = format_bytes(total_selected_bytes)

    # Display total
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        all_selected = len(selected_rows) == len(display_df)
        label_text = (
            f"Total Memory for All {len(selected_rows)} Items"
            if all_selected
            else f"Total Memory for {len(selected_rows)} Selected Items"
        )

        st.metric(
            label=label_text,
            value=total_selected_formatted,
            help=f"Sum of {len(selected_rows)} selected items",
        )

    # Show details
    with st.expander(f"Details of {len(selected_rows)} Selected Items"):
        selected_details = selected_rows[
            ["Module/Group", "Type", "Memory Usage"]
        ].copy()
        st.dataframe(selected_details, hide_index=True, use_container_width=True)


def create_tensor_details_dataframe(tensors: List[Dict]) -> pd.DataFrame:
    """Create a DataFrame with tensor details for display."""
    tensor_details = []

    for tensor in tensors:
        processed_stack = process_stack_trace_for_display(
            tensor.get("create_stack_trace", [])
        )

        tensor_details.append(
            {
                "Size (MB)": tensor.get("size_mb", 0),
                "Size": format_bytes(tensor.get("size_bytes", 0)),
                "Shape": tensor.get("shape", "Unknown"),
                "Module": tensor.get("module", "Unknown"),
                "Phase": tensor.get("phase", "Unknown"),
                "Creation Stack": "\n".join(processed_stack),
            }
        )

    df = pd.DataFrame(tensor_details)
    if "Size (MB)" in df.columns:
        df = df.sort_values("Size (MB)", ascending=False)

    return df


def display_module_node_details(node: Dict) -> None:
    """Display details for a single module node in an expander."""
    if not node["tensors"]:
        return

    # Handle special groups
    if node.get("is_special_group", False):
        st.info(
            f"📊 This group contains {len(node['tensors'])} tensors from the {node['display_path']} phase"
        )

        # Group by module
        module_groups = defaultdict(list)
        for t in node["tensors"]:
            module = t.get("module", "Unknown")
            module_groups[module].append(t)

        st.write(f"**Modules in this group:** {len(module_groups)}")
    else:
        # Check for merged tensors
        sample_tensor = node["tensors"][0] if node["tensors"] else None
        if sample_tensor and sample_tensor.get("_merged", False):
            merged_count = sample_tensor.get("_merged_count", 0)
            merged_pattern = sample_tensor.get("_merged_pattern", "")
            st.info(
                f"⚠️ This represents {merged_count} similar modules matching pattern: {merged_pattern}"
            )

    # Create and display tensor details
    tensor_df = create_tensor_details_dataframe(node["tensors"])

    if not tensor_df.empty:
        column_config = {
            "Size (MB)": st.column_config.NumberColumn(
                "Size (MB)", help="Tensor size in megabytes", format="%.2f"
            ),
            "Creation Stack": st.column_config.TextColumn(
                "Creation Stack",
                help="Stack trace from tensor creation (last 10 lines from megatron/core)",
                width="large",
            ),
        }

        st.dataframe(
            tensor_df,
            column_config=column_config,
            use_container_width=True,
            hide_index=True,
        )


def display_module_hierarchy(flattened_nodes: List[Dict]) -> None:
    """Display the module hierarchy with expandable details."""
    for node in flattened_nodes:
        with st.expander(f"{node['name']} (Path: {node['display_path']})"):
            display_module_node_details(node)


def create_module_tree_chart(device_data: Dict, tensors: List[Dict]) -> None:
    """Create and display complete module tree visualization."""
    module_tree = build_module_tree(tensors)

    # Phase breakdown
    st.header("Memory Analysis")
    phase_groups = group_by_phase(tensors)
    create_phase_breakdown_chart(phase_groups)

    st.divider()

    # Module hierarchy
    st.header("Module Hierarchy")
    flattened_nodes = flatten_module_tree(module_tree)

    # Total memory metric
    total_memory = sum(node["size_bytes"] for node in flattened_nodes)
    st.metric(label="Total Memory in Module Tree", value=format_bytes(total_memory))

    # Merged modules chart
    create_merged_modules_chart(flattened_nodes)

    # Module hierarchy details
    display_module_hierarchy(flattened_nodes)


def display_stack_trace_groups(tensors: List[Dict]) -> None:
    """Display tensors grouped by stack traces."""
    stack_groups = group_by_stack_trace(tensors)

    # Sort by total size
    sorted_groups = sorted(
        stack_groups.items(),
        key=lambda x: sum(t.get("size_bytes", 0) for t in x[1]),
        reverse=True,
    )

    st.header("Tensors Grouped by Creation Stack Trace")

    for i, (stack_key, group_tensors) in enumerate(
        sorted_groups[:MAX_STACK_TRACE_GROUPS]
    ):
        total_size = sum(t.get("size_bytes", 0) for t in group_tensors)

        with st.expander(
            f"Group {i+1}: {len(group_tensors)} tensors, Total: {format_bytes(total_size)}"
        ):
            # Display stack trace preview
            if group_tensors and "create_stack_trace" in group_tensors[0]:
                st.code(
                    "\n".join(
                        group_tensors[0]["create_stack_trace"][
                            :STACK_TRACE_PREVIEW_LINES
                        ]
                    )
                )

            # Display tensor details
            tensor_data = []
            for tensor in group_tensors:
                tensor_data.append(
                    {
                        "Size": format_bytes(tensor.get("size_bytes", 0)),
                        "Shape": tensor.get("shape", "Unknown"),
                        "Module": tensor.get("module", "Unknown").split(".")[-1],
                        "Phase": tensor.get("phase", "Unknown"),
                    }
                )

            tensor_df = pd.DataFrame(tensor_data)
            st.dataframe(tensor_df, hide_index=True)


# ============================================================================
# Command Line and Main Functions
# ============================================================================


@st.cache_data
def parse_command_line_args() -> Optional[str]:
    """Parse command-line arguments without modifying sys.argv."""
    for i in range(1, len(sys.argv) - 1):
        if sys.argv[i] in ["--file", "-f"]:
            return sys.argv[i + 1]
    return None


def initialize_streamlit_page() -> None:
    """Initialize Streamlit page configuration."""
    st.set_page_config(
        page_title="PyTorch Memory Visualizer",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("PyTorch Memory Profiler Visualizer")


def handle_file_loading() -> Optional[Dict]:
    """
    Handle file loading from command line or file uploader.

    Returns:
        Loaded snapshot data or None
    """
    # Initialize session state
    if "snapshot_data" not in st.session_state:
        st.session_state.snapshot_data = None

    # Try command-line file first
    file_arg = parse_command_line_args()
    if file_arg and st.session_state.snapshot_data is None:
        snapshot_data = load_snapshot_from_file(file_arg)
        if snapshot_data:
            st.success(f"Loaded memory snapshot from: {file_arg}")
            st.session_state.snapshot_data = snapshot_data

    # Handle file uploader
    uploaded_file = st.file_uploader(
        "Upload a memory snapshot JSON file", type=["json"]
    )
    if uploaded_file is not None:
        try:
            snapshot_data = json.load(uploaded_file)
            st.session_state.snapshot_data = snapshot_data
        except Exception as e:
            st.error(f"Error loading uploaded file: {e}")

    return st.session_state.snapshot_data


def display_snapshot_data(snapshot_data: Dict) -> None:
    """Display the loaded snapshot data."""
    devices = list(snapshot_data.keys())

    # Device selection
    selected_device = (
        st.selectbox("Select Device", devices) if len(devices) > 1 else devices[0]
    )

    device_data = snapshot_data[selected_device]
    tensors = device_data.get("tensors", [])
    peak_memory_mb = device_data.get("peak_memory_mb", 0)

    st.header(f"Peak Memory: {peak_memory_mb:.2f} MB on {selected_device}")

    # Display module tree visualization
    create_module_tree_chart(device_data, tensors)


def main():
    """Main entry point for the Streamlit application."""
    initialize_streamlit_page()

    snapshot_data = handle_file_loading()

    if snapshot_data:
        display_snapshot_data(snapshot_data)


if __name__ == "__main__":
    main()
