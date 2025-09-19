from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import List, Tuple, override

from minipat.arc import CycleArc, CycleSpan
from minipat.common import CycleTime
from minipat.ev import Ev, EvHeap, ev_heap_empty, ev_heap_push
from minipat.stream import Stream
from spiny import PHeapMap

type Section[T] = Tuple[CycleArc, Stream[T]]


@dataclass(frozen=True)
class ComposeStream[T](Stream[T]):
    """Stream that composes multiple sections with their own timing arcs.

    The compose always starts at CycleTime 0, and section arcs are
    relative to this zero point.
    """

    containing_arc: CycleArc
    sections: List[Tuple[CycleArc, Stream[T]]]

    @override
    def unstream(self, arc: CycleArc) -> PHeapMap[CycleSpan, Ev[T]]:
        if arc.null() or not self.sections:
            return ev_heap_empty()

        result: PHeapMap[CycleSpan, Ev[T]] = ev_heap_empty()

        # Find intersection between query arc and our containing arc
        query_intersection = arc.intersect(self.containing_arc)
        if query_intersection.null():
            return ev_heap_empty()

        # For each section, check if it intersects with the query
        for section_arc, section_stream in self.sections:
            # Find intersection between section arc and query arc
            section_intersection = section_arc.intersect(query_intersection)
            if not section_intersection.null():
                # Always query the substream for a full canonical cycle to get all events
                canonical_cycle = CycleArc(
                    CycleTime(Fraction(0)), CycleTime(Fraction(1))
                )
                section_events = section_stream.unstream(canonical_cycle)

                # Calculate the shifted intersection for clipping
                shifted_start = section_intersection.start - section_arc.start
                shifted_end = section_intersection.end - section_arc.start
                shifted_intersection = CycleArc(
                    CycleTime(shifted_start), CycleTime(shifted_end)
                )

                # Filter and shift events back to the section's time window
                for _, ev in section_events:
                    # Check if this event intersects with the shifted intersection
                    event_intersection = ev.span.active.intersect(shifted_intersection)
                    if not event_intersection.null():
                        # Calculate the active arc (clipped to the intersection)
                        active_start = event_intersection.start + section_arc.start
                        active_end = event_intersection.end + section_arc.start
                        active_arc = CycleArc(
                            CycleTime(active_start), CycleTime(active_end)
                        )

                        # Calculate the whole arc (full event extent in section time)
                        whole_start = ev.span.active.start + section_arc.start
                        whole_end = ev.span.active.end + section_arc.start
                        whole_arc = CycleArc(
                            CycleTime(whole_start), CycleTime(whole_end)
                        )

                        # Only set whole if it's different from active (i.e., the event was clipped)
                        final_whole = whole_arc if whole_arc != active_arc else None

                        span = CycleSpan(active_arc, final_whole)
                        shifted_ev = Ev(span, ev.val)
                        result = ev_heap_push(shifted_ev, result)

        return result


def compose[T](sections: List[Section[T]]) -> EvHeap[T]:
    """Compose a song by rendering sections.

    This function takes a list of (CycleArc, Stream) pairs and creates a
    composed stream. The compose starts at CycleTime 0 and the section
    arcs are relative to this zero point.

    Args:
        sections: A list of (arc, stream) pairs where arc indicates when
                and for how long the stream should play, relative to time 0.

    Returns:
        A Stream[T] representing the composed result

    """
    if not sections:
        return ev_heap_empty()

    # Sort sections by their arc start time to ensure chronological ordering
    sorted_sections = sorted(sections, key=lambda section: section[0].start)

    # Calculate the containing arc (0 to max end time of all sections)
    max_end = max(section_arc.end for section_arc, _ in sorted_sections)
    containing_arc = CycleArc(CycleTime(Fraction(0)), CycleTime(max_end))

    # Create a compose stream that switches between patterns based on timing
    stream = ComposeStream(containing_arc, sorted_sections)

    return stream.unstream(containing_arc)
