"""Graph-of-graphs result extraction for the MetaLead LaTeX set.

Top graph: papers, weighted by embedding similarity and shared names, grouped with Louvain.
Inner graph: chunks (tables and sections) of the papers in one group, linked across papers.
Extraction reads each chunk with its closest chunks from other papers as reference context,
then results that several papers report corroborate each other.
"""
