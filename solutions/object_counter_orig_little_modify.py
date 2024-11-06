# Ultralytics YOLO 🚀, AGPL-3.0 license

from collections import defaultdict

import cv2

from ultralytics.utils.checks import check_imshow, check_requirements
from ultralytics.utils.plotting import Annotator, colors

check_requirements("shapely>=2.0.0")

from shapely.geometry import LineString, Point, Polygon


class ObjectCounter:
    """A class to manage the counting of objects in a real-time video stream based on their tracks."""

    def __init__(
        self,
        classes_names,
        reg_pts=None,
        count_reg_color=(255, 0, 255),
        count_txt_color=(0, 0, 0),
        count_bg_color=(255, 255, 255),
        line_thickness=2,
        track_thickness=2,
        view_img=False,
        view_in_counts=True,
        view_out_counts=True,
        draw_tracks=False,
        track_color=None,
        region_thickness=5,
        line_dist_thresh=15,
        cls_txtdisplay_gap=50,
    ):
        """
        Initializes the ObjectCounter with various tracking and counting parameters.

        Args:
            classes_names (dict): Dictionary of class names.
            reg_pts (list): List of points defining the counting region.
            count_reg_color (tuple): RGB color of the counting region.
            count_txt_color (tuple): RGB color of the count text.
            count_bg_color (tuple): RGB color of the count text background.
            line_thickness (int): Line thickness for bounding boxes.
            track_thickness (int): Thickness of the track lines.
            view_img (bool): Flag to control whether to display the video stream.
            view_in_counts (bool): Flag to control whether to display the in counts on the video stream.
            view_out_counts (bool): Flag to control whether to display the out counts on the video stream.
            draw_tracks (bool): Flag to control whether to draw the object tracks.
            track_color (tuple): RGB color of the tracks.
            region_thickness (int): Thickness of the object counting region.
            line_dist_thresh (int): Euclidean distance threshold for line counter.
            cls_txtdisplay_gap (int): Display gap between each class count.
        """

        # Mouse events
        self.is_drawing = False
        self.selected_point = None

        # Region & Line Information
        self.reg_pts = [(20, 400), (1260, 400)] if reg_pts is None else reg_pts
        self.line_dist_thresh = line_dist_thresh
        self.counting_region = None
        self.region_color = count_reg_color
        self.region_thickness = region_thickness

        # Image and annotation Information
        self.im0 = None
        self.tf = line_thickness
        self.view_img = view_img
        self.view_in_counts = view_in_counts
        self.view_out_counts = view_out_counts

        self.names = classes_names  # Classes names
        self.annotator = None  # Annotator
        self.window_name = "Ultralytics YOLOv8 Object Counter"

        # Object counting Information
        self.in_counts = 0
        self.out_counts = 0
        self.count_ids = []
        self.class_wise_count = {}
        self.count_txt_thickness = 0
        self.count_txt_color = count_txt_color
        self.count_bg_color = count_bg_color
        self.cls_txtdisplay_gap = cls_txtdisplay_gap
        self.fontsize = 0.6

        # Tracks info
        self.track_history = defaultdict(list)
        self.track_thickness = track_thickness
        self.draw_tracks = draw_tracks
        self.track_color = track_color

        # Check if environment supports imshow
        self.env_check = check_imshow(warn=True)

        # Initialize counting region
        if len(self.reg_pts) == 2:
            print("Line Counter Initiated.")
            self.counting_region = LineString(self.reg_pts)
        elif len(self.reg_pts) >= 3:
            print("Polygon Counter Initiated.")
            self.counting_region = Polygon(self.reg_pts)
        else:
            print("Invalid Region points provided, region_points must be 2 for lines or >= 3 for polygons.")
            print("Using Line Counter Now")
            self.counting_region = LineString(self.reg_pts)

    def mouse_event_for_region(self, event, x, y, flags, params):
        """
        Handles mouse events for defining and moving the counting region in a real-time video stream.

        Args:
            event (int): The type of mouse event (e.g., cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN, etc.).
            x (int): The x-coordinate of the mouse pointer.
            y (int): The y-coordinate of the mouse pointer.
            flags (int): Any associated event flags (e.g., cv2.EVENT_FLAG_CTRLKEY,  cv2.EVENT_FLAG_SHIFTKEY, etc.).
            params (dict): Additional parameters for the function.
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            for i, point in enumerate(self.reg_pts):
                if (
                    isinstance(point, (tuple, list))
                    and len(point) >= 2
                    and (abs(x - point[0]) < 10 and abs(y - point[1]) < 10)
                ):
                    self.selected_point = i
                    self.is_drawing = True
                    break

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.is_drawing and self.selected_point is not None:
                self.reg_pts[self.selected_point] = (x, y)
                self.counting_region = Polygon(self.reg_pts)

        elif event == cv2.EVENT_LBUTTONUP:
            self.is_drawing = False
            self.selected_point = None

    def extract_and_process_tracks(self, tracks):
        """Extracts and processes tracks for object counting in a video stream."""

        # Annotator Init and region drawing
        self.annotator = Annotator(self.im0, self.tf, self.names)

        # Draw region or line
        self.annotator.draw_region(reg_pts=self.reg_pts, color=self.region_color, thickness=self.region_thickness)

        if tracks[0].boxes.id is not None:
            boxes = tracks[0].boxes.xyxy.cpu()
            clss = tracks[0].boxes.cls.cpu().tolist()
            track_ids = tracks[0].boxes.id.int().cpu().tolist()

            # Extract tracks
            for box, track_id, cls in zip(boxes, track_ids, clss):
                # Draw bounding box
                self.annotator.box_label(box, label=f"{self.names[cls]}#{track_id}", color=colors(int(track_id), True))

                # Store class info
                if self.names[cls] not in self.class_wise_count:
                    self.class_wise_count[self.names[cls]] = {"IN": 0, "OUT": 0}

                # Draw Tracks
                track_line = self.track_history[track_id]
                track_line.append((float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2)))
                if len(track_line) > 30:
                    track_line.pop(0)

                # Draw track trails
                if self.draw_tracks:
                    self.annotator.draw_centroid_and_tracks(
                        track_line,
                        color=self.track_color if self.track_color else colors(int(track_id), True),
                        track_thickness=self.track_thickness,
                    )

                prev_position = self.track_history[track_id][-2] if len(self.track_history[track_id]) > 1 else None

                # Count objects in any polygon
                if len(self.reg_pts) >= 3:
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
                elif len(self.reg_pts) == 2:
                    if prev_position is not None and track_id not in self.count_ids:
                        distance = Point(track_line[-1]).distance(self.counting_region)
                        if distance < self.line_dist_thresh and track_id not in self.count_ids:
                            self.count_ids.append(track_id)

                            if (box[0] - prev_position[0]) * (self.counting_region.centroid.x - prev_position[0]) > 0:
                                self.in_counts += 1
                                self.class_wise_count[self.names[cls]]["IN"] += 1
                            else:
                                self.out_counts += 1
                                self.class_wise_count[self.names[cls]]["OUT"] += 1

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
            self.annotator.display_analytics(self.im0, labels_dict, self.count_txt_color, self.count_bg_color, 10)

    def display_frames(self):
        """Displays the current frame with annotations and regions in a window."""
        if self.env_check:
            cv2.namedWindow(self.window_name)
            if len(self.reg_pts) == 4:  # only add mouse event If user drawn region
                cv2.setMouseCallback(self.window_name, self.mouse_event_for_region, {"region_points": self.reg_pts})
            cv2.imshow(self.window_name, self.im0)
            # Break Window
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return

    def start_counting(self, im0, tracks):
        """
        Main function to start the object counting process.

        Args:
            im0 (ndarray): Current frame from the video stream.
            tracks (list): List of tracks obtained from the object tracking process.
        """
        self.im0 = im0  # store image
        self.extract_and_process_tracks(tracks)  # draw region even if no objects

        if self.view_img:
            self.display_frames()
        return self.im0


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
    ObjectCounter(classes_names)
