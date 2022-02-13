# -*- coding: utf-8 -*-

from pathlib import Path
import pickle

"""
Loading Morphology file desrcibing structure of the neuron and coordinates
of different branches.
"""
morphologyFilePath = Path("./files/morphology.pickle")
morphologyFile = morphologyFilePath.open(mode="rb")
morphologyDict = pickle.load(morphologyFile, encoding='latin1')

segmentsLength = morphologyDict["all_segments_length"]
segmentDistFromSoma = morphologyDict['all_sections_distance_from_soma']
segmentsType = morphologyDict['all_segments_type']
segmentsSectionDistFromSoma = morphologyDict['all_segments_section_distance_from_soma']
segmentsSectionInd = morphologyDict['all_segments_section_index']
segmentIndexWithinSectionIndex = morphologyDict[
    'all_segments_segment_index_within_section_index']

basalSectionCoords = morphologyDict['all_basal_section_coords']
basalSegmentCoords = morphologyDict['all_basal_segment_coords']
apicalSectionCoords = morphologyDict['all_apical_section_coords']
apicalSegmentCoords = morphologyDict['all_apical_segment_coords']

segmentIdxToCoords = {}
segmentIdxToSectionIdx = {}

for idx, segment in enumerate(segmentsType[:5]):

    currentSegmentIdx = segmentIndexWithinSectionIndex[idx]
    if segment == "basal":
        currentSectionIdx = segmentsSectionInd[idx]
        segmentCoords = basalSegmentCoords[(currentSectionIdx, currentSegmentIdx)]
        segmentIdxToCoords[idx] = segmentCoords
        segmentIdxToSectionIdx[idx] = ("basal", currentSectionIdx)

    if segment == "apical":
        currentSectionIdx = segmentsSectionInd[idx] - len(basalSectionCoords)
        segmentCoords = apicalSegmentCoords[(currentSectionIdx, currentSegmentIdx)]
        segmentIdxToCoords[idx] = segmentCoords
        segmentIdxToSectionIdx[idx] = ("apical", currentSectionIdx)
