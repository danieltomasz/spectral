
def define_region_dict():
    
    roi_names = [
        "bankssts L",
        "bankssts R",
        "caudalanteriorcingulate L",
        "caudalanteriorcingulate R",
        "caudalmiddlefrontal L",
        "caudalmiddlefrontal R",
        "cuneus L",
        "cuneus R",
        "entorhinal L",
        "entorhinal R",
        "frontalpole L",
        "frontalpole R",
        "fusiform L",
        "fusiform R",
        "inferiorparietal L",
        "inferiorparietal R",
        "inferiortemporal L",
        "inferiortemporal R",
        "insula L",
        "insula R",
        "isthmuscingulate L",
        "isthmuscingulate R",
        "lateraloccipital L",
        "lateraloccipital R",
        "lateralorbitofrontal L",
        "lateralorbitofrontal R",
        "lingual L",
        "lingual R",
        "medialorbitofrontal L",
        "medialorbitofrontal R",
        "middletemporal L",
        "middletemporal R",
        "paracentral L",
        "paracentral R",
        "parahippocampal L",
        "parahippocampal R",
        "parsopercularis L",
        "parsopercularis R",
        "parsorbitalis L",
        "parsorbitalis R",
        "parstriangularis L",
        "parstriangularis R",
        "pericalcarine L",
        "pericalcarine R",
        "postcentral L",
        "postcentral R",
        "posteriorcingulate L",
        "posteriorcingulate R",
        "precentral L",
        "precentral R",
        "precuneus L",
        "precuneus R",
        "rostralanteriorcingulate L",
        "rostralanteriorcingulate R",
        "rostralmiddlefrontal L",
        "rostralmiddlefrontal R",
        "superiorfrontal L",
        "superiorfrontal R",
        "superiorparietal L",
        "superiorparietal R",
        "superiortemporal L",
        "superiortemporal R",
        "supramarginal L",
        "supramarginal R",
        "temporalpole L",
        "temporalpole R",
        "transversetemporal L",
        "transversetemporal R",
    ]

    frontal = [
        "caudalmiddlefrontal",
        "frontalpole",
        "lateralorbitofrontal",
        "medialorbitofrontal",
        "paracentral",
        "parsopercularis",
        "parsorbitalis",
        "parstriangularis",
        "precentral",
        "rostralmiddlefrontal",
        "superiorfrontal",
    ]
    temporal = [
        "bankssts",
        "entorhinal",
        "fusiform",
        "inferiortemporal",
        "middletemporal",
        "parahippocampal",
        "superiortemporal",
        "temporalpole",
        "transversetemporal",
    ]
    cingulate = [
        "caudalanteriorcingulate",
        "insula",
        "isthmuscingulate",
        "posteriorcingulate",
        "rostralanteriorcingulate",
    ]
    occipital = ["cuneus", "lateraloccipital", "lingual", "pericalcarine", "precuneus"]
    parietal = ["inferiorparietal", "postcentral", "superiorparietal", "supramarginal"]

    return {
        "temporal": [r for r in roi_names for f in temporal if f in r],
        "frontal": [r for r in roi_names for f in frontal if f in r],
        "cingulate": [r for r in roi_names for f in cingulate if f in r],
        "occipital": [r for r in roi_names for f in occipital if f in r],
        "parietal": [r for r in roi_names for f in parietal if f in r],
    }

def prepare_grouping(da, coordinate: str,  partition_dict: dict):
    return [  # assing specific coordinates
        key
        for element in da[coordinate].values
        for key, value in partition_dict.items()
        if element in value
    ]
