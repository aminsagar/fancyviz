import pandas as pd
import numpy as np
import re
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pyvis.network import Network
import json
import warnings
warnings.filterwarnings('ignore')

def parse_motif(motif):
    """
    Parse a motif to extract its components.
    Returns the base pattern with brackets and the surrounding amino acids.
    """
    # Find the bracketed pattern
    bracket_match = re.search(r'([A-Z]*)\[(\d+)\]([A-Z]*)', motif)
    if bracket_match:
        before = bracket_match.group(1)
        number = bracket_match.group(2)
        after = bracket_match.group(3)
        core = f"{before[-1] if before else ''}[{number}]{after[0] if after else ''}"
        prefix = before[:-1] if len(before) > 1 else ""
        suffix = after[1:] if len(after) > 1 else ""
        return {
            'core': core,
            'prefix': prefix,
            'suffix': suffix,
            'full': motif,
            'number': int(number)
        }
    return None

def is_parent_of(parent_motif, child_motif):
    """
    Check if parent_motif is a parent of child_motif.
    A parent is a substring that maintains the bracket structure.
    """
    parent_parsed = parse_motif(parent_motif)
    child_parsed = parse_motif(child_motif)
    
    if not parent_parsed or not child_parsed:
        return False
    
    # Must have the same bracket number
    if parent_parsed['number'] != child_parsed['number']:
        return False
    
    # Check if parent is contained in child
    # This means child has all the components of parent plus potentially more
    if parent_motif in child_motif:
        # Additional check: make sure the bracket structure is preserved
        if f"[{parent_parsed['number']}]" in child_motif:
            return True
    
    return False

def find_motif_hierarchy(motifs_df):
    """
    Build a hierarchy of motifs based on parent-child relationships.
    """
    motifs = motifs_df['core_pattern'].tolist()
    hierarchy = defaultdict(list)
    parent_map = {}
    
    # Sort motifs by length (shorter ones are more likely to be parents)
    motifs_sorted = sorted(motifs, key=len)
    
    for i, potential_parent in enumerate(motifs_sorted):
        for j, potential_child in enumerate(motifs_sorted):
            if i != j and is_parent_of(potential_parent, potential_child):
                hierarchy[potential_parent].append(potential_child)
                # Keep track of the most immediate parent
                if potential_child not in parent_map or len(potential_parent) > len(parent_map[potential_child]):
                    parent_map[potential_child] = potential_parent
    
    return hierarchy, parent_map

def consolidate_counts(motifs_df, hierarchy):
    """
    Consolidate counts from children to parents.
    """
    consolidated = motifs_df.copy()
    consolidated['original_count'] = consolidated['total_sequences']
    consolidated['child_motifs'] = ''
    consolidated['num_children'] = 0
    
    # Create a mapping of motif to its row index
    motif_to_idx = {row['core_pattern']: idx for idx, row in consolidated.iterrows()}
    
    # For each parent, add counts from all its descendants
    def get_all_descendants(motif, hierarchy, visited=None):
        if visited is None:
            visited = set()
        if motif in visited:
            return []
        visited.add(motif)
        
        descendants = []
        if motif in hierarchy:
            for child in hierarchy[motif]:
                descendants.append(child)
                descendants.extend(get_all_descendants(child, hierarchy, visited))
        return descendants
    
    for parent, children in hierarchy.items():
        if parent in motif_to_idx:
            parent_idx = motif_to_idx[parent]
            all_descendants = get_all_descendants(parent, hierarchy)
            
            # Remove duplicates while preserving order
            seen = set()
            unique_descendants = []
            for desc in all_descendants:
                if desc not in seen and desc in motif_to_idx:
                    seen.add(desc)
                    unique_descendants.append(desc)
            
            # Sum up counts from all descendants
            child_count_sum = 0
            for child in unique_descendants:
                if child in motif_to_idx:
                    child_idx = motif_to_idx[child]
                    child_count_sum += consolidated.loc[child_idx, 'original_count']
            
            consolidated.loc[parent_idx, 'total_sequences'] = (
                consolidated.loc[parent_idx, 'original_count'] + child_count_sum
            )
            consolidated.loc[parent_idx, 'child_motifs'] = ', '.join(unique_descendants[:10])  # Show first 10
            consolidated.loc[parent_idx, 'num_children'] = len(unique_descendants)
    
    return consolidated

def create_interactive_network_pyvis(hierarchy, motifs_df, save_path=None):
    """
    Create an interactive network visualization using PyVis.
    """
    # Create network
    net = Network(height='900px', width='100%', 
                  bgcolor='#222222', font_color='white',
                  directed=True)
    
    # Configure physics
    net.barnes_hut(gravity=-5000, central_gravity=0.3, spring_length=200)
    net.show_buttons(filter_=['physics'])
    
    # Create motif info dictionary
    motif_info = {}
    for _, row in motifs_df.iterrows():
        motif = row['core_pattern']
        motif_info[motif] = {
            'count': row['total_sequences'],
            'original': row.get('original_count', row['total_sequences']),
            'children': row.get('num_children', 0)
        }
    
    # Find root nodes
    all_children = set()
    for children in hierarchy.values():
        all_children.update(children)
    root_nodes = set(motif_info.keys()) - all_children
    
    # Calculate hierarchy levels
    def get_level(node, hierarchy, level_cache={}):
        if node in level_cache:
            return level_cache[node]
        
        if node in root_nodes:
            level = 0
        else:
            # Find parent
            parent = None
            for p, children in hierarchy.items():
                if node in children:
                    parent = p
                    break
            if parent:
                level = get_level(parent, hierarchy, level_cache) + 1
            else:
                level = 0
        
        level_cache[node] = level
        return level
    
    # Add nodes
    for motif, info in motif_info.items():
        count = info['count']
        original = info['original']
        children = info['children']
        
        # Size based on count (log scale)
        size = max(10, min(50, np.log1p(count) * 5))
        
        # Color based on hierarchy level
        level = get_level(motif, hierarchy)
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57', '#ff9ff3']
        color = colors[level % len(colors)]
        
        # Create hover text
        title = (f"<b>{motif}</b><br>"
                f"Total: {count:,}<br>"
                f"Original: {original:,}<br>"
                f"Children: {children}<br>"
                f"Level: {level}")
        
        net.add_node(motif, 
                    label=motif,
                    title=title,
                    size=size,
                    color=color,
                    level=level,
                    font={'size': max(8, min(14, size/3))})
    
    # Add edges (only direct parent-child relationships)
    for parent, children in hierarchy.items():
        for child in children:
            # Check if this is a direct relationship (no intermediate parent)
            is_direct = True
            for other_parent in hierarchy:
                if other_parent != parent and child in hierarchy.get(other_parent, []):
                    if is_parent_of(other_parent, parent):
                        is_direct = False
                        break
            
            if is_direct:
                net.add_edge(parent, child, arrows='to', color='#666666')
    
    # Save network
    if save_path:
        net.save_graph(save_path)
        print(f"Interactive network saved to {save_path}")
    return net

def create_interactive_plotly_tree(hierarchy, motifs_df, save_prefix=None):
    """
    Create an interactive tree/sunburst visualization using Plotly.
    """
    # Prepare data for treemap/sunburst
    labels = []
    parents = []
    values = []
    colors = []
    hover_text = []
    
    # Create motif info dictionary
    motif_info = {}
    for _, row in motifs_df.iterrows():
        motif = row['core_pattern']
        motif_info[motif] = {
            'count': row['total_sequences'],
            'original': row.get('original_count', row['total_sequences']),
            'children': row.get('num_children', 0)
        }
    
    # Find root nodes
    all_children = set()
    for children_list in hierarchy.values():
        all_children.update(children_list)
    root_nodes = set(motif_info.keys()) - all_children
    
    # Add root node
    labels.append("All Motifs")
    parents.append("")
    values.append(0)  # Will be calculated by Plotly
    colors.append(0)
    hover_text.append("Root")
    
    # Add all motifs
    for motif, info in motif_info.items():
        labels.append(motif)
        
        # Find parent
        parent_motif = None
        for p, children in hierarchy.items():
            if motif in children:
                # Find the most immediate parent
                if parent_motif is None or len(p) > len(parent_motif):
                    parent_motif = p
        
        if parent_motif:
            parents.append(parent_motif)
        else:
            parents.append("All Motifs")
        
        values.append(info['original'])  # Use original count to avoid double counting
        colors.append(info['count'])
        hover_text.append(f"Total: {info['count']:,}<br>Original: {info['original']:,}<br>Children: {info['children']}")
    
    # Create sunburst chart
    fig_sunburst = go.Figure(go.Sunburst(
        labels=labels,
        parents=parents,
        values=values,
        text=labels,
        hovertemplate='<b>%{label}</b><br>%{customdata}<extra></extra>',
        customdata=hover_text,
        marker=dict(
            colors=colors,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Total Count")
        ),
        branchvalues="total"
    ))
    
    fig_sunburst.update_layout(
        title="Motif Hierarchy - Interactive Sunburst (Click to zoom)",
        width=1000,
        height=1000,
        template='plotly_dark'
    )
    
    # Create treemap
    fig_treemap = go.Figure(go.Treemap(
        labels=labels,
        parents=parents,
        values=values,
        text=labels,
        hovertemplate='<b>%{label}</b><br>%{customdata}<extra></extra>',
        customdata=hover_text,
        marker=dict(
            colors=colors,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Total Count")
        ),
        textposition="middle center",
        pathbar=dict(visible=True)
    ))
    
    fig_treemap.update_layout(
        title="Motif Hierarchy - Interactive Treemap (Click to drill down)",
        width=1200,
        height=800,
        template='plotly_dark'
    )
    
    # Save both visualizations
    if save_prefix:
        fig_sunburst.write_html(f"{save_prefix}_sunburst.html")
        fig_treemap.write_html(f"{save_prefix}_treemap.html")
        print(f"Interactive sunburst saved to {save_prefix}_sunburst.html")
        print(f"Interactive treemap saved to {save_prefix}_treemap.html")
    
    return fig_sunburst, fig_treemap

def create_interactive_scatter(consolidated_df, save_path=None):
    """
    Create an interactive scatter plot showing original vs consolidated counts.
    """
    # Prepare data
    df_plot = consolidated_df.copy()
    df_plot['count_increase'] = df_plot['total_sequences'] - df_plot['original_count']
    df_plot['increase_pct'] = (df_plot['count_increase'] / df_plot['original_count'].replace(0, 1)) * 100
    
    # Create scatter plot
    fig = go.Figure()
    
    # Add scatter trace
    fig.add_trace(go.Scatter(
        x=df_plot['original_count'],
        y=df_plot['total_sequences'],
        mode='markers',
        marker=dict(
            size=np.log1p(df_plot['num_children']) * 5 + 5,
            color=df_plot['num_children'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="# Children"),
            line=dict(width=1, color='white')
        ),
        text=df_plot['core_pattern'],
        hovertemplate=(
            '<b>%{text}</b><br>' +
            'Original: %{x:,}<br>' +
            'Consolidated: %{y:,}<br>' +
            'Children: %{marker.color}<br>' +
            '<extra></extra>'
        )
    ))
    
    # Add diagonal line
    max_val = max(df_plot['total_sequences'].max(), df_plot['original_count'].max())
    fig.add_trace(go.Scatter(
        x=[0, max_val],
        y=[0, max_val],
        mode='lines',
        line=dict(color='gray', dash='dash'),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        title="Original vs Consolidated Counts (Interactive)",
        xaxis_title="Original Count",
        yaxis_title="Consolidated Count",
        template='plotly_dark',
        width=900,
        height=700,
        xaxis_type="log",
        yaxis_type="log"
    )
    
    if save_path:
        fig.write_html(save_path)
        print(f"Interactive scatter plot saved to {save_path}")
    return fig

def create_dashboard(consolidated_df, hierarchy, save_path=None):
    """
    Create a comprehensive interactive dashboard.
    """
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Top 20 Motifs by Consolidated Count', 
                       'Count Increase Distribution',
                       'Children Distribution', 
                       'Motif Length vs Count'),
        specs=[[{'type': 'bar'}, {'type': 'histogram'}],
               [{'type': 'bar'}, {'type': 'scatter'}]]
    )
    
    # Sort by consolidated count
    df_sorted = consolidated_df.sort_values('total_sequences', ascending=False)
    
    # 1. Top motifs
    top_20 = df_sorted.head(20)
    fig.add_trace(
        go.Bar(x=top_20['core_pattern'], 
               y=top_20['total_sequences'],
               name='Consolidated',
               marker_color='lightblue',
               hovertemplate='%{x}<br>Count: %{y:,}<extra></extra>'),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(x=top_20['core_pattern'], 
               y=top_20['original_count'],
               name='Original',
               marker_color='coral',
               hovertemplate='%{x}<br>Count: %{y:,}<extra></extra>'),
        row=1, col=1
    )
    
    # 2. Count increase distribution
    count_increase_pct = ((df_sorted['total_sequences'] - df_sorted['original_count']) / 
                         df_sorted['original_count'].replace(0, 1) * 100)
    count_increase_pct = count_increase_pct[count_increase_pct > 0]
    
    fig.add_trace(
        go.Histogram(x=count_increase_pct,
                    nbinsx=30,
                    marker_color='green',
                    name='Count Increase %',
                    hovertemplate='Increase: %{x:.1f}%<br>Count: %{y}<extra></extra>'),
        row=1, col=2
    )
    
    # 3. Children distribution
    children_counts = df_sorted['num_children'].value_counts().sort_index()
    fig.add_trace(
        go.Bar(x=children_counts.index,
               y=children_counts.values,
               marker_color='purple',
               name='# Motifs',
               hovertemplate='Children: %{x}<br>Motifs: %{y}<extra></extra>'),
        row=2, col=1
    )
    
    # 4. Motif length vs count
    df_sorted['motif_length'] = df_sorted['core_pattern'].str.len()
    fig.add_trace(
        go.Scatter(x=df_sorted['motif_length'],
                  y=df_sorted['total_sequences'],
                  mode='markers',
                  marker=dict(
                      size=np.log1p(df_sorted['num_children']) * 3 + 3,
                      color=df_sorted['num_children'],
                      colorscale='Viridis',
                      showscale=True,
                      colorbar=dict(title="Children", x=1.15)
                  ),
                  text=df_sorted['core_pattern'],
                  name='Motifs',
                  hovertemplate='%{text}<br>Length: %{x}<br>Count: %{y:,}<extra></extra>'),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title_text="Motif Consolidation Dashboard",
        showlegend=True,
        template='plotly_dark',
        height=800,
        width=1400
    )
    
    # Update axes
    fig.update_xaxes(title_text="Motif", row=1, col=1, tickangle=45)
    fig.update_xaxes(title_text="Count Increase (%)", row=1, col=2)
    fig.update_xaxes(title_text="Number of Children", row=2, col=1)
    fig.update_xaxes(title_text="Motif Length", row=2, col=2)
    
    fig.update_yaxes(title_text="Count", row=1, col=1, type='log')
    fig.update_yaxes(title_text="Number of Motifs", row=1, col=2)
    fig.update_yaxes(title_text="Number of Motifs", row=2, col=1)
    fig.update_yaxes(title_text="Consolidated Count", row=2, col=2, type='log')
    
    if save_path:
        fig.write_html(save_path)
        print(f"Interactive dashboard saved to {save_path}")
    return fig

def create_static_summary(original_df, consolidated_df, save_path=None):
    """
    Create static summary statistics visualization (kept from original).
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Top motifs comparison
    ax = axes[0, 0]
    top_n = 15
    top_consolidated = consolidated_df.nlargest(top_n, 'total_sequences')
    
    y_pos = np.arange(len(top_consolidated))
    ax.barh(y_pos, top_consolidated['total_sequences'], alpha=0.7, label='Consolidated')
    ax.barh(y_pos, top_consolidated['original_count'], alpha=0.7, label='Original')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_consolidated['core_pattern'])
    ax.set_xlabel('Total Sequences')
    ax.set_title(f'Top {top_n} Motifs - Original vs Consolidated Counts')
    ax.legend()
    ax.invert_yaxis()
    
    # 2. Distribution of child counts
    ax = axes[0, 1]
    child_counts = consolidated_df['num_children'].value_counts().sort_index()
    ax.bar(child_counts.index, child_counts.values, alpha=0.7, color='green')
    ax.set_xlabel('Number of Children')
    ax.set_ylabel('Number of Parent Motifs')
    ax.set_title('Distribution of Child Motifs per Parent')
    ax.grid(True, alpha=0.3)
    
    # 3. Count increase distribution
    ax = axes[1, 0]
    count_increase = consolidated_df['total_sequences'] - consolidated_df['original_count']
    count_increase_pct = (count_increase / consolidated_df['original_count'].replace(0, 1)) * 100
    
    ax.hist(count_increase_pct[count_increase_pct > 0], bins=30, alpha=0.7, color='orange')
    ax.set_xlabel('Count Increase (%)')
    ax.set_ylabel('Number of Motifs')
    ax.set_title('Distribution of Count Increases from Consolidation')
    ax.grid(True, alpha=0.3)
    
    # 4. Summary statistics table
    ax = axes[1, 1]
    ax.axis('tight')
    ax.axis('off')
    
    summary_data = [
        ['Metric', 'Original', 'Consolidated'],
        ['Total Motifs', len(original_df), len(consolidated_df)],
        ['Total Sequences (sum)', f"{original_df['total_sequences'].sum():,}", 
         f"{consolidated_df['total_sequences'].sum():,}"],
        ['Max Count', f"{original_df['total_sequences'].max():,}", 
         f"{consolidated_df['total_sequences'].max():,}"],
        ['Mean Count', f"{original_df['total_sequences'].mean():.1f}", 
         f"{consolidated_df['total_sequences'].mean():.1f}"],
        ['Motifs with Children', '-', 
         f"{(consolidated_df['num_children'] > 0).sum()}"],
        ['Max Children per Motif', '-', 
         f"{consolidated_df['num_children'].max()}"]
    ]
    
    table = ax.table(cellText=summary_data, loc='center', cellLoc='left')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Style the header row
    for i in range(3):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax.set_title('Summary Statistics', fontsize=12, fontweight='bold', pad=20)
    
    plt.suptitle('Motif Consolidation Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Static summary saved to {save_path}")
    plt.show()

def analyze_motifs(df, generate_figures=True):
    """
    Run the motif consolidation pipeline on a dataframe and optionally prepare figures.
    Returns a dictionary containing the inputs, intermediary data, and any generated figures.
    """
    hierarchy, parent_map = find_motif_hierarchy(df)
    consolidated_df = consolidate_counts(df, hierarchy)
    consolidated_df = consolidated_df.sort_values('total_sequences', ascending=False).reset_index(drop=True)
    
    results = {
        'original_df': df,
        'consolidated_df': consolidated_df,
        'hierarchy': hierarchy,
        'parent_map': parent_map
    }
    
    if generate_figures:
        sunburst_fig, treemap_fig = create_interactive_plotly_tree(hierarchy, consolidated_df, save_prefix=None)
        results['figures'] = {
            'sunburst': sunburst_fig,
            'treemap': treemap_fig,
            'scatter': create_interactive_scatter(consolidated_df, save_path=None),
            'dashboard': create_dashboard(consolidated_df, hierarchy, save_path=None),
            'network': create_interactive_network_pyvis(hierarchy, consolidated_df, save_path=None)
        }
    
    return results

def main():
    # Read the CSV file
    print("Reading CSV file...")
    df = pd.read_csv('seq_relative_p70_enhanced_motifs.csv')
    
    print(f"Loaded {len(df)} motifs")
    print(f"Total sequences before consolidation: {df['total_sequences'].sum():,}")
    
    # Find motif hierarchy
    print("\nBuilding motif hierarchy...")
    hierarchy, parent_map = find_motif_hierarchy(df)
    
    print(f"Found {len(hierarchy)} parent motifs")
    print(f"Found {len(parent_map)} child-parent relationships")
    
    # Consolidate counts
    print("\nConsolidating counts...")
    consolidated_df = consolidate_counts(df, hierarchy)
    
    print(f"Total sequences after consolidation: {consolidated_df['total_sequences'].sum():,}")
    
    # Save consolidated data
    output_file = 'consolidated_motifs.csv'
    consolidated_df.sort_values('total_sequences', ascending=False, inplace=True)
    consolidated_df.to_csv(output_file, index=False)
    print(f"\nSaved consolidated data to {output_file}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # Interactive network visualization with PyVis
    print("  - Creating interactive network (PyVis)...")
    create_interactive_network_pyvis(hierarchy, consolidated_df, 'motif_network_interactive.html')
    
    # Interactive tree visualizations with Plotly
    print("  - Creating interactive tree visualizations (Plotly)...")
    create_interactive_plotly_tree(hierarchy, consolidated_df, save_prefix='motif_tree_interactive')
    
    # Interactive scatter plot
    print("  - Creating interactive scatter plot...")
    create_interactive_scatter(consolidated_df, 'motif_scatter_interactive.html')
    
    # Interactive dashboard
    print("  - Creating interactive dashboard...")
    create_dashboard(consolidated_df, hierarchy, 'motif_dashboard.html')
    
    # Static summary (kept from original)
    print("  - Creating static summary...")
    create_static_summary(df, consolidated_df, 'motif_summary.png')
    
    # Print top consolidated motifs
    print("\nTop 10 Consolidated Motifs:")
    print("-" * 80)
    top_10 = consolidated_df.head(10)
    for idx, row in top_10.iterrows():
        print(f"{row['core_pattern']:20} | Original: {row['original_count']:6,} | "
              f"Consolidated: {row['total_sequences']:6,} | Children: {row['num_children']:3}")
    
    print("\n" + "="*80)
    print("Analysis complete! Generated files:")
    print("="*80)
    print("\nDATA FILES:")
    print("  📊 consolidated_motifs.csv - Consolidated data with parent-child info")
    
    print("\nINTERACTIVE VISUALIZATIONS (open in browser):")
    print("  🌐 motif_network_interactive.html - Draggable network with physics simulation")
    print("  🌻 motif_tree_interactive_sunburst.html - Zoomable sunburst chart")
    print("  📦 motif_tree_interactive_treemap.html - Drillable treemap")
    print("  📈 motif_scatter_interactive.html - Interactive scatter plot")
    print("  📊 motif_dashboard.html - Comprehensive dashboard")
    
    print("\nSTATIC VISUALIZATION:")
    print("  📸 motif_summary.png - Summary statistics")
    
    print("\n💡 TIP: The interactive visualizations allow you to:")
    print("  • Zoom, pan, and explore the data")
    print("  • Click on nodes to expand/collapse branches")
    print("  • Hover to see detailed information")
    print("  • Filter and adjust the physics in the network view")

if __name__ == "__main__":
    main()
