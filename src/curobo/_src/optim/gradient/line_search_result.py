from dataclasses import dataclass
from .line_search_state import LineSearchState
@dataclass
class LineSearchResult:
    selected_state: LineSearchState
    exploration_state: LineSearchState
