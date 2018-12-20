import xml.etree.ElementTree as ET


class BoundingBox(object):
    pass

def GetItem(name, root, index=0):
    count = 0
    for item in root.iter(name):
        if count == index:
            return item.text
        count += 1
    # Failed to find "index" occurrence of item.
    return -1

def GetInt(name, root, index=0):
    return int(GetItem(name, root, index))

def FindNumberBoundingBoxes(root):
    index = 0
    while True:
        if GetInt('xmin', root, index) == -1:
            break
        index += 1
    return index

def retrieve_bounding_box(xml_file):
    """Process a single XML file containing a bounding box."""
    # pylint: disable=broad-except
    try:
        tree = ET.parse(xml_file)
    except Exception:
        print('Failed to parse: ' + xml_file, file=sys.stderr)
        return None
    # pylint: enable=broad-except
    root = tree.getroot()

    num_boxes = FindNumberBoundingBoxes(root)
    boxes = []
    
    for index in range(num_boxes):
        box = BoundingBox()
        # Grab the 'index' annotation.
        box.xmin = GetInt('xmin', root, index)
        box.ymin = GetInt('ymin', root, index)
        box.xmax = GetInt('xmax', root, index)
        box.ymax = GetInt('ymax', root, index)

        box.width = GetInt('width', root)
        box.height = GetInt('height', root)
        box.filename = GetItem('filename', root) + '.JPEG'
        box.label = GetItem('name', root)

        #xmin = float(box.xmin) / float(box.width)
        #xmax = float(box.xmax) / float(box.width)
        #ymin = float(box.ymin) / float(box.height)
        #ymax = float(box.ymax) / float(box.height)

        # Some images contain bounding box annotations that
        # extend outside of the supplied image. See, e.g.
        # n03127925/n03127925_147.xml
        # Additionally, for some bounding boxes, the min > max
        # or the box is entirely outside of the image.
        #min_x = min(xmin, xmax)
        #max_x = max(xmin, xmax)
        #box.xmin_scaled = min(max(min_x, 0.0), 1.0)
        #box.xmax_scaled = min(max(max_x, 0.0), 1.0)

        #min_y = min(ymin, ymax)
        #max_y = max(ymin, ymax)
        #box.ymin_scaled = min(max(min_y, 0.0), 1.0)
        #box.ymax_scaled = min(max(max_y, 0.0), 1.0)
        box.xmin_scaled = max(0, box.xmin)
        box.xmax_scaled = min(box.width, box.xmax)
  
        box.ymin_scaled = max(0, box.ymin)
        box.ymax_scaled = min(box.height, box.ymax)
        boxes.append([(box.xmin_scaled, box.ymin_scaled), (box.xmax_scaled, box.ymax_scaled)])

    return boxes


