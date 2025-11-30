import cv2
import numpy as np

class RadarView:
    def __init__(self):
        # Configuration for the Radar View
        self.court_width_meters = 9
        self.court_length_meters = 18
        self.free_zone_meters = 2 # Margin around the court
        
        # Pixel scale (pixels per meter)
        self.pixels_per_meter = 40 
        
        # Calculate dimensions
        self.total_width_meters = self.court_width_meters + (2 * self.free_zone_meters)
        self.total_length_meters = self.court_length_meters + (2 * self.free_zone_meters)
        
        self.img_width = int(self.total_width_meters * self.pixels_per_meter)
        self.img_height = int(self.total_length_meters * self.pixels_per_meter)
        
        # Calculate offsets for the playing court within the image
        self.margin_x = int(self.free_zone_meters * self.pixels_per_meter)
        self.margin_y = int(self.free_zone_meters * self.pixels_per_meter)
        
        # Store Homography Matrix for player projection
        self.M = None

    def _order_points(self, pts):
        """
        Orders coordinates in the order: [top-left, top-right, bottom-right, bottom-left]
        """
        rect = np.zeros((4, 2), dtype="float32")
        pts = np.array(pts, dtype="float32")

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)] # TL
        rect[2] = pts[np.argmax(s)] # BR

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)] # TR
        rect[3] = pts[np.argmax(diff)] # BL

        return rect

    def get_warped_frame(self, frame, points):
        """
        Generates a synthetic radar view of the court.
        Calculates the homography matrix M for future tracking projections.
        """
        if not points or len(points) < 4:
            return None

        # 1. Calculate Homography Matrix (needed for future tracking)
        src_pts = self._order_points(points[:4])
        
        # Destination points correspond to the corners of the PLAYING COURT (inside the margins)
        dst_pts = np.array([
            [self.margin_x, self.margin_y],                                      # TL
            [self.img_width - self.margin_x, self.margin_y],                     # TR
            [self.img_width - self.margin_x, self.img_height - self.margin_y],   # BR
            [self.margin_x, self.img_height - self.margin_y]                     # BL
        ], dtype="float32")

        self.M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        
        # 2. Create Synthetic Background
        # Create blank image (Blue background for Free Zone)
        # Color is BGR: (255, 144, 30) -> Blue-ish
        radar_img = np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)
        radar_img[:] = (255, 144, 30) 
        
        # Draw Playing Court Area (Orange)
        # Color BGR: (100, 180, 255) -> Orange-ish
        cv2.rectangle(radar_img, 
                      (self.margin_x, self.margin_y), 
                      (self.img_width - self.margin_x, self.img_height - self.margin_y), 
                      (100, 180, 255), -1) # -1 fills the rectangle
        
        # 3. Draw Overlay Lines
        radar_img = self.draw_court_overlay(radar_img)
        
        return radar_img

    def draw_court_overlay(self, img):
        """
        Draws the standard volleyball lines and net onto the warped image.
        """
        # Colors (BGR)
        line_color = (255, 255, 255) # White
        net_color = (50, 50, 50)     # Dark Gray for net visual
        
        thickness = 2
        
        # 1. Draw Perimeter
        tl = (self.margin_x, self.margin_y)
        br = (self.img_width - self.margin_x, self.img_height - self.margin_y)
        cv2.rectangle(img, tl, br, line_color, thickness)
        
        # 2. Draw Center Line (Net)
        center_y = self.img_height // 2
        # Draw line across the whole width (including free zone) or just court? 
        # Usually poles are outside, so let's draw slightly outside
        cv2.line(img, (self.margin_x - 20, center_y), (self.img_width - self.margin_x + 20, center_y), net_color, 4)
        
        # 3. Draw Attack Lines (3m lines)
        # 3 meters in pixels
        attack_line_px = int(3 * self.pixels_per_meter)
        
        # Top Attack Line
        cv2.line(img, 
                 (self.margin_x, center_y - attack_line_px), 
                 (self.img_width - self.margin_x, center_y - attack_line_px), 
                 line_color, thickness)
        
        # Bottom Attack Line
        cv2.line(img, 
                 (self.margin_x, center_y + attack_line_px), 
                 (self.img_width - self.margin_x, center_y + attack_line_px), 
                 line_color, thickness)

        # 4. Draw Service Zone Ticks (Optional - visual cue for extensions)
        tick_len = 10
        # Top extension
        cv2.line(img, (self.margin_x, self.margin_y), (self.margin_x, self.margin_y - tick_len), line_color, thickness)
        cv2.line(img, (self.img_width - self.margin_x, self.margin_y), (self.img_width - self.margin_x, self.margin_y - tick_len), line_color, thickness)
        # Bottom extension
        cv2.line(img, (self.margin_x, self.img_height - self.margin_y), (self.margin_x, self.img_height - self.margin_y + tick_len), line_color, thickness)
        cv2.line(img, (self.img_width - self.margin_x, self.img_height - self.margin_y), (self.img_width - self.margin_x, self.img_height - self.margin_y + tick_len), line_color, thickness)

        return img