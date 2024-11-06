# Ultralytics YOLO ??, AGPL-3.0 license

from collections import defaultdict
import cv2
import numpy as np
from ultralytics.utils.checks import check_imshow, check_requirements
from ultralytics.utils.plotting import Annotator
from shapely.geometry import LineString, Point, Polygon

check_requirements("shapely>=2.0.0")

class Heatmap:
    """A class to draw heatmaps in real-time video stream based on their tracks."""

    def __init__(
        self,
        classes_names,
        imw=0,
        imh=0,
        colormap=cv2.COLORMAP_JET,
        heatmap_alpha=0.5,
        view_img=False,
        view_in_counts=True,
        view_out_counts=True,
        count_reg_pts=None,
        count_txt_color=(0, 0, 0),
        count_bg_color=(255, 255, 255),
        count_reg_color=(255, 0, 255),
        region_thickness=5,
        line_dist_thresh=15,
        line_thickness=2,
        decay_factor=0.99,
        shape="circle",
    ):
        """Initializes the heatmap class with default values for Visual, Image, track, count and heatmap parameters."""

        # Visual information
        self.annotator = None
        self.view_img = view_img
        self.shape = shape

        self.initialized = False
        self.names = classes_names  # Classes names

        # Image information
        self.imw = imw
        self.imh = imh
        self.im0 = None
        self.tf = line_thickness
        self.view_in_counts = view_in_counts
        self.view_out_counts = view_out_counts

        # Heatmap colormap and heatmap np array
        self.colormap = colormap
        self.heatmap = None
        self.heatmap_alpha = heatmap_alpha

        # Predict/track information
        self.boxes = None
        self.track_ids = None
        self.clss = None
        self.track_history = defaultdict(list)

        # Region & Line Information
        self.counting_region = None
        self.line_dist_thresh = line_dist_thresh
        self.region_thickness = region_thickness
        self.region_color = count_reg_color

        # Object Counting Information
        self.in_counts = 0
        self.out_counts = 0
        self.count_ids = []
        self.class_wise_count = {}
        self.count_txt_color = count_txt_color
        self.count_bg_color = count_bg_color
        self.cls_txtdisplay_gap = 50

        # Decay factor
        self.decay_factor = decay_factor

        # Check if environment supports imshow
        self.env_check = check_imshow(warn=True)

        # Region and line selection
        self.count_reg_pts = count_reg_pts
        print(self.count_reg_pts)
        if self.count_reg_pts is not None:
            if len(self.count_reg_pts) == 2:
                print("Line Counter Initiated.")
                self.counting_region = LineString(self.count_reg_pts)
            elif len(self.count_reg_pts) >= 3:
                print("Polygon Counter Initiated.")
                self.counting_region = Polygon(self.count_reg_pts)
            else:
                print("Invalid Region points provided, region_points must be 2 for lines or >= 3 for polygons.")
                print("Using Line Counter Now")
                self.counting_region = LineString(self.count_reg_pts)

        # Shape of heatmap, if not selected
        if self.shape not in {"circle", "rect"}:
            print("Unknown shape value provided, 'circle' & 'rect' supported")
            print("Using Circular shape now")
            self.shape = "circle"

    def extract_results(self, tracks, _intialized=False):
        """
        Extracts results from the provided data.

        Args:
            tracks (list): List of tracks obtained from the object tracking process.
        """
        self.boxes = tracks[0].boxes.xyxy.cpu()
        self.clss = tracks[0].boxes.cls.cpu().tolist()
        if tracks and tracks[0].boxes and tracks[0].boxes.id is not None:
            self.track_ids = tracks[0].boxes.id.int().cpu().tolist()
        else:
            print("No valid track IDs found for heatmap generation.")

    def generate_heatmap(self, im0, tracks):
        """
        Generate heatmap based on tracking data.

        Args:
            im0 (nd array): Image
            tracks (list): List of tracks obtained from the object tracking process.
        """
        self.im0 = im0

        # Initialize heatmap only once
        if not self.initialized:
            self.heatmap = np.zeros((int(self.im0.shape[0]), int(self.im0.shape[1])), dtype=np.float32)
            self.initialized = True

        self.heatmap *= self.decay_factor  # decay factor

        self.extract_results(tracks)
        self.annotator = Annotator(self.im0, self.tf, None)

        if self.track_ids is not None:
            # Draw counting region
            if self.count_reg_pts is not None:
                self.annotator.draw_region(
                    reg_pts=self.count_reg_pts, color=self.region_color, thickness=self.region_thickness
                )

            for box, cls, track_id in zip(self.boxes, self.clss, self.track_ids):
                # Store class info
                if self.names[cls] not in self.class_wise_count:
                    self.class_wise_count[self.names[cls]] = {"IN": 0, "OUT": 0}

                # Draw fish shapes using bounding boxes
                self.heatmap[int(box[1]):int(box[3]), int(box[0]):int(box[2])] += 2

                # Optionally, draw contours or use masks for more detailed shapes if available
                if hasattr(tracks[0].boxes, 'masks') and tracks[0].boxes.masks is not None:
                    mask = tracks[0].boxes.masks[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                    self.heatmap[int(box[1]):int(box[3]), int(box[0]):int(box[2])] += 2 * mask

                # Store tracking history
                track_line = self.track_history[track_id]
                track_line.append((float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2)))
                if len(track_line) > 30:
                    track_line.pop(0)

                prev_position = self.track_history[track_id][-2] if len(self.track_history[track_id]) > 1 else None

                if self.count_reg_pts is not None:
                    # Count objects in any polygon
                    if len(self.count_reg_pts) >= 3:
                        is_inside = self.counting_region.contains(Point(track_line[-1]))

                        if prev_position is not None and is_inside and track_id not in self.count_ids:
                            self.count_ids.append(track_id)

                            if (box[0] - prev_position[0]) * (self.counting_region.centroid.x - prev_position[0]) > 0:
                                self.in_counts += 1
                                self.class_wise_count[self.names[cls]]["IN"] += 1
                            else:
                                self.out_counts += 1
                                self.class_wise_count[self.names[cls]]["OUT"] += 1

                    # Count objects using line
                    elif len(self.count_reg_pts) == 2:
                        if prev_position is not None and track_id not in self.count_ids:
                            distance = Point(track_line[-1]).distance(self.counting_region)
                            if distance < self.line_dist_thresh and track_id not in self.count_ids:
                                self.count_ids.append(track_id)

                                if (box[0] - prev_position[0]) * (
                                    self.counting_region.centroid.x - prev_position[0]
                                ) > 0:
                                    self.in_counts += 1
                                    self.class_wise_count[self.names[cls]]["IN"] += 1
                                else:
                                    self.out_counts += 1
                                    self.class_wise_count[self.names[cls]]["OUT"] += 1

        else:
            for box, cls in zip(self.boxes, self.clss):
                # Draw fish shapes using bounding boxes
                self.heatmap[int(box[1]):int(box[3]), int(box[0]):int(box[2])] += 2

                # Optionally, draw contours or use masks for more detailed shapes if available
                if hasattr(tracks[0].boxes, 'masks') and tracks[0].boxes.masks is not None:
                    mask = tracks[0].boxes.masks[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                    self.heatmap[int(box[1]):int(box[3]), int(box[0]):int(box[2])] += 2 * mask

        if self.count_reg_pts is not None:
            labels_dict = {}

            for key, value in self.class_wise_count.items():
                if value["IN"] != 0 or value["OUT"] != 0:
                    if not self.view_in_counts and not self.view_out_counts:
                        continue
                    elif not self.view_in_counts:
                        labels_dict[str.capitalize(key)] = f"OUT {value['OUT']}"
                    elif not self.view_out_counts:
                        labels_dict[str.capitalize(key)] = f"IN {value['IN']}"
                    else:
                        labels_dict[str.capitalize(key)] = f"IN {value['IN']} OUT {value['OUT']}"

            if labels_dict:
                self.annotator.draw_labels(labels_dict, self.count_txt_color, self.count_bg_color, self.cls_txtdisplay_gap)

        # Apply colormap
        heatmap_rgb = cv2.applyColorMap(np.uint8(self.heatmap), self.colormap)
        heatmap_rgb = cv2.addWeighted(heatmap_rgb, self.heatmap_alpha, self.im0, 1.0, 0)

        return heatmap_rgb

    def resize(self, im0):
        """
        Resize the image based on new dimensions and update heatmap.

        Args:
            im0 (nd array): Image to resize.
        """
        self.im0 = im0
        self.heatmap = np.zeros((int(self.im0.shape[0]), int(self.im0.shape[1])), dtype=np.float32)

    def draw_labels(self, labels_dict, txt_color, bg_color, txt_gap):
        """
        Draw labels with count information on the image.

        Args:
            labels_dict (dict): Dictionary containing labels and their count information.
            txt_color (tuple): Text color.
            bg_color (tuple): Background color.
            txt_gap (int): Gap between text labels.
        """
        y0, dy = 30, 30
        for label in labels_dict:
            color = bg_color
            cv2.putText(
                self.im0,
                labels_dict[label],
                (10, y0),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2,
                cv2.LINE_AA
            )
            y0 += dy

    def display_frames(self):
        """Display frame."""
        cv2.imshow("Ultralytics Heatmap", self.im0)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyAllWindows()

if __name__ == "__main__":
    #classes_names = {0: "person", 1: "car"}  # example class names
    classes_names = {
    0: "ACANTHURUS-170160100",
    1: "ACANTHURUSCOERULEUS-170160102",
    2: "ALECTISCILIARIS-170110101",
    3: "ANISOTREMUSVIRGINICUS-170190105",
    4: "ANOMURA-999100401",
    5: "ARCHOSARGUSPROBATOCEPHALUS-170213601",
    6: "BALISTESCAPRISCUS-189030502",
    7: "BALISTESVETULA-189030504",
    8: "BODIANUSPULCHELLUS-170280201",
    9: "BODIANUSRUFUS-170280202",
    10: "CALAMUS-170210600",
    11: "CALAMUSBAJONADO-170210602",
    12: "CALAMUSLEUCOSTEUS-170210604",
    13: "CALAMUSNODOSUS-170210608",
    14: "CALAMUSPRORIDENS-170210605",
    15: "CALLIONYMIDAE-170420000",
    16: "CANTHIDERMISSUFFLAMEN-189030402",
    17: "CANTHIGASTERROSTRATA-189080101",
    18: "CARANGIDAE-170110000",
    19: "CARANXBARTHOLOMAEI-170110801",
    20: "CARANXCRYSOS-170110803",
    21: "CARANXRUBER-170110807",
    22: "CARCHARHINUS-108020200",
    23: "CARCHARHINUSFALCIFORMIS-108020202",
    24: "CARCHARHINUSPEREZI-108020211",
    25: "CARCHARHINUSPLUMBEUS-108020208",
    26: "CAULOLATILUSCHRYSOPS-170070104",
    27: "CAULOLATILUSCYANOPS-170070101",
    28: "CAULOLATILUSMICROPS-170070103",
    29: "CENTROPRISTISOCYURUS-170024804",
    30: "CENTROPRISTISPHILADELPHICA-170024805",
    31: "CEPHALOPHOLISCRUENTATA-170020401",
    32: "CEPHALOPHOLISFULVA-170020403",
    33: "CHAETODON-170260300",
    34: "CHAETODONCAPISTRATUS-170260302",
    35: "CHAETODONOCELLATUS-170260307",
    36: "CHAETODONSEDENTARIUS-170260309",
    37: "CHROMIS-170270300",
    38: "CHROMISENCHRYSURUS-170270302",
    39: "CHROMISINSOLATUS-170270304",
    40: "DECAPTERUS-170111200",
    41: "DERMATOLEPISINERMIS-170020301",
    42: "DIODONTIDAE-189090000",
    43: "DIPLECTRUM-170020900",
    44: "DIPLECTRUMFORMOSUM-170020903",
    45: "EPINEPHELUS-170021200",
    46: "EPINEPHELUSADSCENSIONIS-170021203",
    47: "EPINEPHELUSMORIO-170021211",
    48: "EQUETUSLANCEOLATUS-170201104",
    49: "GOBIIDAE-170550000",
    50: "GONIOPLECTRUSHISPANUS-170021403",
    51: "GYMNOTHORAXMORINGA-143060202",
    52: "GYMNOTHORAXSAXICOLA-143060205",
    53: "HAEMULONALBUM-170191002",
    54: "HAEMULONAUROLINEATUM-170191003",
    55: "HAEMULONFLAVOLINEATUM-170191005",
    56: "HAEMULONMACROSTOMUM-170191017",
    57: "HAEMULONMELANURUM-170191007",
    58: "HAEMULONPLUMIERI-170191008",
    59: "HALICHOERES-170281200",
    60: "HALICHOERESBATHYPHILUS-170281201",
    61: "HALICHOERESBIVITTATUS-170281202",
    62: "HALICHOERESGARNOTI-170281205",
    63: "HOLACANTHUS-170290100",
    64: "HOLACANTHUSBERMUDENSIS-170290102",
    65: "HOLOCENTRUS-161110200",
    66: "HOLOCENTRUSADSCENSIONIS-161110201",
    67: "HYPOPLECTRUS-170021500",
    68: "HYPOPLECTRUSGEMMA-170021503",
    69: "HYPOPLECTRUSUNICOLOR-170021501",
    70: "HYPORTHODUSFLAVOLIMBATUS-170021206",
    71: "HYPORTHODUSNIGRITUS-170021202",
    72: "IOGLOSSUS-170550800",
    73: "KYPHOSUS-170240300",
    74: "LACHNOLAIMUSMAXIMUS-170281801",
    75: "LACTOPHRYSTRIGONUS-189070205",
    76: "LIOPROPOMAEUKRINES-170025602",
    77: "LUTJANUS-170151100",
    78: "LUTJANUSANALIS-170151101",
    79: "LUTJANUSAPODUS-170151102",
    80: "LUTJANUSBUCCANELLA-170151106",
    81: "LUTJANUSCAMPECHANUS-170151107",
    82: "LUTJANUSGRISEUS-170151109",
    83: "LUTJANUSSYNAGRIS-170151113",
    84: "LUTJANUSVIVANUS-170151114",
    85: "MALACANTHUSPLUMIERI-170070301",
    86: "MULLOIDICHTHYSMARTINICUS-170220101",
    87: "MURAENARETIFERA-143060402",
    88: "MYCTEROPERCA-170022100",
    89: "MYCTEROPERCABONACI-170022101",
    90: "MYCTEROPERCAINTERSTITIALIS-170022103",
    91: "MYCTEROPERCAMICROLEPIS-170022104",
    92: "MYCTEROPERCAPHENAX-170022105",
    93: "OCYURUSCHRYSURUS-170151501",
    94: "OPHICHTHUSPUNCTICEPS-143150402",
    95: "OPISTOGNATHUS-170310200",
    96: "OPISTOGNATHUSAURIFRONS-170310203",
    97: "PAGRUSPAGRUS-170212302",
    98: "PARANTHIASFURCIFER-170022701",
    99: "PAREQUESUMBROSUS-170201105",
    100: "POMACANTHUS-170290200",
    101: "POMACANTHUSARCUATUS-170290201",
    102: "POMACANTHUSPARU-170290203",
    103: "POMACENTRIDAE-170270000",
    104: "POMACENTRUS-170270500",
    105: "POMACENTRUSPARTITUS-170270502",
    106: "PRIACANTHUSARENATUS-170050101",
    107: "PRISTIGENYSALTA-170050401",
    108: "PRISTIPOMOIDES-170151800",
    109: "PROGNATHODESACULEATUS-170260305",
    110: "PROGNATHODESAYA-170260301",
    111: "PRONOTOGRAMMUSMARTINICENSIS-170025101",
    112: "PSEUDUPENEUSMACULATUS-170220701",
    113: "PTEROIS-168011900",
    114: "RACHYCENTRONCANADUM-170100101",
    115: "RHOMBOPLITESAURORUBENS-170152001",
    116: "RYPTICUSMACULATUS-170030106",
    117: "SCARIDAE-170300000",
    118: "SCARUSVETULA-170301107",
    119: "SCOMBEROMORUS-170440800",
    120: "SERIOLA-170113100",
    121: "SERIOLADUMERILI-170113101",
    122: "SERIOLAFASCIATA-170113103",
    123: "SERIOLARIVOLIANA-170113105",
    124: "SERIOLAZONATA-170113106",
    125: "SERRANUS-170024200",
    126: "SERRANUSANNULARIS-170024201",
    127: "SERRANUSATROBRANCHUS-170024202",
    128: "SERRANUSPHOEBE-170024208",
    129: "SPARIDAE-170210000",
    130: "SPARISOMAAUROFRENATUM-170301201",
    131: "SPARISOMAVIRIDE-170301206",
    132: "SPHYRAENABARRACUDA-165030101",
    133: "SPHYRNALEWINI-108040102",
    134: "STENOTOMUSCAPRINUS-170213403",
    135: "SYACIUM-183011000",
    136: "SYNODONTIDAE-129040000",
    137: "THALASSOMABIFASCIATUM-170282801",
    138: "UNKNOWNFISH",
    139: "UPENEUSPARVUS-170220605",
    140: "UROPHYCISREGIA-148010105",
    141: "XANTHICHTHYSRINGENS-189030101"
    }
    heatmap = Heatmap(classes_names)
