# Required dataset entry keys
_PREFIX = 'image_directory'
_SOURCE_INDEX = 'image_and_label_list_file'
_MEAN = 'rgb_mean'
_STD = 'rgb_std'
_CLASSES_LIST = 'class_names_list'
_NUM_CLASSES = 'identities number'
_VIS_COLORS = 'classes_colors_for_vis'
_DATASETS = {
    'whu_opt_sar256': {
        _PREFIX: '/home/wty/data/whu_opt_sar/whu_opt_sar_crop256',
        _SOURCE_INDEX: {
            'train': 'datasets/whu_opt_sar256/train_60.txt',
            'val': 'datasets/whu_opt_sar256/val_20.txt',
            'test': 'datasets/whu_opt_sar256/test_20.txt'
        },
        _MEAN: [104.6953, 96.7183, 79.1122, 100.7773, 54.0023],
        _STD: [14.6102, 16.5178, 19.7112, 29.9144, 48.2651],
        _NUM_CLASSES: 8,
        _VIS_COLORS: {
            'background': [0, 0, 0],
            'farmland': [204, 102, 1],
            'city': [254, 0, 0],
            'village': [255, 255, 1],
            'water': [0, 0, 254],
            'forest': [85, 166, 1],
            'road': [93, 255, 255],
            'others': [152, 102, 153],
        }
    },
    'dfc20': {
        _PREFIX: '/home/wty/data/DFC2020',
        _SOURCE_INDEX: {
            'train': 'datasets/dfc2020/train_60.txt',
            'val': 'datasets/dfc2020/val_20.txt',
            'test': 'datasets/dfc2020/test_20.txt',
        },
        # 3 band optical 2 band SAR
        _MEAN: [763.8806, 919.1531, 1013.227, -12.542, -19.7742],
        _STD: [248.1907, 173.0708, 141.8084, 2.4201, 2.5095],
        _NUM_CLASSES: 9,
        _VIS_COLORS: {
            'background': [0, 0, 0],
            'Forests': [0, 128, 0],
            'Shrublands': [0, 51, 0],
            'Grasslands': [173, 255, 47],
            'Wetlands': [205, 133, 63],
            'Croplands': [245, 222, 179],
            'Urban_build_up': [112, 128, 144],
            'Barren': [210, 180, 140],
            'Water': [0, 0, 128],
        },
    },
    'C2Seg_BW_80': {
        # 3 band optical 2 band SAR
        _PREFIX: '/home/wty/data/C2Seg/C2Seg_BW/train/',
        _SOURCE_INDEX: {
            'train': 'datasets/c2seg_bw/train_80.txt',
            'val': 'datasets/c2seg_bw/val_80.txt',
            'test': 'datasets/c2seg_bw/val_80.txt'
        },
        # 4 band optical 2 band SAR
        _MEAN: [511.2993, 664.0968, 769.0629, 1248.1037, -15.968, -24.2466],
        _STD: [511.2993, 664.0968, 769.0629, 1248.1037, 4.6059, 4.1042],
        _NUM_CLASSES: 14,
        _VIS_COLORS: {
            'Background': [0, 0, 0],
            'Surface water': [0, 255, 255],
            'Street': [255, 255, 255],
            'Urban Fabric': [255, 0, 0],
            'Industrial, commercial and transport': [220, 160, 220],
            'Mine, dump, and construction sites': [150, 0, 210],
            'Artificial, vegetated areas': [255, 130, 255],
            'Arable Land': [255, 220, 130],
            'Permanent Crops': [206, 133, 64],
            'Pastures': [189, 183, 107],
            'Forests': [0, 255, 0],
            'Shrub': [154, 205, 50],
            'Open spaces with no vegetation': [139, 69, 18],
            'Inland wetlands': [130, 111, 255]
        },
    },
    'none': {
        _PREFIX:
            [''],
        _MEAN:
            [0.5, 0.5, 0.5],
        _STD:
            [0.5, 0.5, 0.5],
        _NUM_CLASSES: 1000
    },
}
_LABEL_MAPPING = 'label_mapping'


# Available datasets

def datasets():
    """Retrieve the list of available dataset names."""
    return _DATASETS.keys()


def contains(name):
    return name in _DATASETS.keys()


def get_prefix(name):
    return _DATASETS[name][_PREFIX]


def get_source_index(name):
    return _DATASETS[name][_SOURCE_INDEX]


def get_num_classes(name):
    return _DATASETS[name][_NUM_CLASSES]


def get_mean(name):
    return _DATASETS[name][_MEAN]


def get_std(name):
    return _DATASETS[name][_STD]


def get_names_list(name):
    return _DATASETS[name][_CLASSES_LIST]


def get_vis_colors(name):
    return _DATASETS[name][_VIS_COLORS]


def get_label_mapping(name):
    return _DATASETS[name][_LABEL_MAPPING]
