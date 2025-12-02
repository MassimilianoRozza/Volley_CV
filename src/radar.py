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
        
        # Active tracking zone: 'all', 'left', 'right'
        self.active_zone = 'all'
        
        # Orientation: 'vertical' (Behind Baseline), 'horizontal' (Sideline)
        self.orientation = 'vertical'
        
        # Invert Sides Flag
        self.invert_sides = False
        # Button rect: x, y, w, h
        self.button_rect = (10, 10, 120, 30)
        
        # Mirror Left-Right Flag
        self.mirror_lr = False
        # Mirror Button rect: x, y, w, h
        self.mirror_button_rect = (140, 10, 120, 30)
        
        # Cache for static court image
        self.static_court_img = None

    def set_active_zone(self, zone):
        """
        Sets the active tracking zone.
        zone: 'all', 'left', 'right'
        """
        self.active_zone = zone

    def set_orientation(self, orientation):
        """
        Sets the court orientation.
        orientation: 'vertical' or 'horizontal'
        """
        self.orientation = orientation
        
    def draw_buttons(self, img):
        """
        Draws the interface buttons on the radar view.
        """
        # 1. Swap Sides Button
        x, y, w, h = self.button_rect
        color = (0, 255, 0) if self.invert_sides else (200, 200, 200)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)
        cv2.rectangle(img, (x, y), (x + w, y + h), (50, 50, 50), 2)
        cv2.putText(img, "SWAP SIDES", (x + 10, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # 2. Mirror LR Button
        mx, my, mw, mh = self.mirror_button_rect
        m_color = (0, 255, 0) if self.mirror_lr else (200, 200, 200)
        cv2.rectangle(img, (mx, my), (mx + mw, my + mh), m_color, -1)
        cv2.rectangle(img, (mx, my), (mx + mw, my + mh), (50, 50, 50), 2)
        cv2.putText(img, "MIRROR LR", (mx + 10, my + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return img

    def check_button_click(self, x, y):
        """
        Checks if any interface button was clicked.
        Returns True if state changed.
        """
        # Check Swap Sides
        bx, by, bw, bh = self.button_rect
        if bx <= x <= bx + bw and by <= y <= by + bh:
            self.invert_sides = not self.invert_sides
            return True
            
        # Check Mirror LR
        mx, my, mw, mh = self.mirror_button_rect
        if mx <= x <= mx + mw and my <= y <= my + mh:
            self.mirror_lr = not self.mirror_lr
            return True
            
        return False

    def _order_points(self, pts):
        """
        Orders a list of 4 points in the order: [top-left, top-right, bottom-right, and bottom-left].
        This is crucial for consistent perspective transform.
        """
        pts = np.array(pts, dtype="float32")
        
        # Sort the points based on their x-coordinates
        x_sorted = pts[np.argsort(pts[:, 0]), :]

        # Grab the left-most and right-most points from the sorted
        # x-coordinate points
        left_most = x_sorted[:2, :]
        right_most = x_sorted[2:, :]

        # Now, sort the left-most coordinates by their y-coordinates so we can
        # grab the top-left and bottom-left
        left_most = left_most[np.argsort(left_most[:, 1]), :]
        (tl, bl) = left_most

        # Do the same for the right-most points
        right_most = right_most[np.argsort(right_most[:, 1]), :]
        (tr, br) = right_most

        # Return the coordinates in the order of top-left, top-right,
        # bottom-right, and bottom-left
        return np.array([tl, tr, br, bl], dtype="float32")

    def _draw_static_court(self):
        """
        Helper to draw the static background and lines of the radar view.
        Uses caching to avoid redrawing every frame.
        """
        if self.static_court_img is not None:
            return self.static_court_img.copy()

        # 1. Create Synthetic Background
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
        
        # Draw Overlay Lines
        radar_img = self.draw_court_overlay(radar_img)
        
        self.static_court_img = radar_img
        return radar_img.copy()

    def update_homography(self, points):
        """
        Calculates and updates the homography matrix M based on the provided court points.
        """
        if not points or len(points) < 4:
            return

        src_pts = self._order_points(points[:4])
        
        # Define Corner Coordinates of the Radar Court (inside margins)
        # Standard Vertical Layout:
        # TL(0,0) -------- TR(W,0)
        # |                  |
        # |                  |
        # BL(0,H) -------- BR(W,H)
        
        tl_dst = [self.margin_x, self.margin_y]
        tr_dst = [self.img_width - self.margin_x, self.margin_y]
        br_dst = [self.img_width - self.margin_x, self.img_height - self.margin_y]
        bl_dst = [self.margin_x, self.img_height - self.margin_y]

        if self.orientation == 'horizontal':
            # Sideline View (Landscape Video) -> Vertical Radar
            # We map the Screen Rect (Long width) to the Radar Rect (Long Height)
            # Rotation 90 deg counter-clockwise mapping:
            # Screen TL (Src 0) -> Radar TL
            # Screen TR (Src 1) -> Radar BL (Left Edge of Radar becomes Top Edge of Screen)
            # Screen BR (Src 2) -> Radar BR
            # Screen BL (Src 3) -> Radar TR
            
            dst_pts = np.array([
                tl_dst, # Src TL -> Dst TL
                bl_dst, # Src TR -> Dst BL (Down the left side)
                br_dst, # Src BR -> Dst BR
                tr_dst  # Src BL -> Dst TR
            ], dtype="float32")
            
        else:
            # Vertical View (Baseline View) - Standard
            dst_pts = np.array([
                tl_dst,
                tr_dst,
                br_dst,
                bl_dst
            ], dtype="float32")

        self.M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    def is_in_bounds(self, image_point):
        """
        Checks if a point (x, y) in the original image space corresponds to a location
        within the defined radar view (Court + Free Zone) AND matches the active zone.
        """
        if self.M is None:
            return True # If no calibration, assume everything is valid to avoid breaking tracking

        x, y = image_point
        
        # Transform point
        # cv2.perspectiveTransform expects shape (N, 1, 2)
        pt = np.array([[[x, y]]], dtype="float32")
        dst_pt = cv2.perspectiveTransform(pt, self.M)
        
        dx = dst_pt[0][0][0]
        dy = dst_pt[0][0][1]
        
        # 1. Check Global Bounds (Radar Image Dimensions)
        if not ((0 <= dx < self.img_width) and (0 <= dy < self.img_height)):
            return False
            
        # 2. Check Active Zone
        center_x = self.img_width / 2
        center_y = self.img_height / 2
        
        if self.orientation == 'horizontal':
            # Sideline View Mapping
            # Screen Left -> Radar Top (0 to H/2)
            # Screen Right -> Radar Bottom (H/2 to H)
            
            if self.active_zone == 'left':
                # User wants Left Half of Screen -> Keep Radar Top Half
                if dy >= center_y: return False
            elif self.active_zone == 'right':
                # User wants Right Half of Screen -> Keep Radar Bottom Half
                if dy <= center_y: return False
        else:
            # Standard Vertical Mapping
            if self.active_zone == 'left':
                if dx >= center_x:
                    return False
            elif self.active_zone == 'right':
                if dx <= center_x:
                    return False
                
        return True

    def get_radar_guide(self, phase_idx, point_idx):
        """
        Generates a static radar view with a visual cue (flashing dot or highlight)
        indicating which point the user should select next.
        
        phase_idx: 
            0 = Perimeter (4 pts)
            1 = Near 3m (2 pts)
            2 = Net (2 pts)
            3 = Far 3m (2 pts)
        point_idx: The index of the point within the phase (0, 1, 2, 3...)
        """
        img = self._draw_static_court()
        
        target_x, target_y = -1, -1
        
        center_y = self.img_height // 2
        attack_line_px = int(3 * self.pixels_per_meter)
        
        # Define target coordinates based on phase and point index
        if phase_idx == 0: # Perimeter: TL, TR, BR, BL (Standard reading order for guidance)
            if point_idx == 0:   # TL
                target_x, target_y = self.margin_x, self.margin_y
            elif point_idx == 1: # TR
                target_x, target_y = self.img_width - self.margin_x, self.margin_y
            elif point_idx == 2: # BR
                target_x, target_y = self.img_width - self.margin_x, self.img_height - self.margin_y
            elif point_idx == 3: # BL
                target_x, target_y = self.margin_x, self.img_height - self.margin_y
                
        elif phase_idx == 1: # Near 3m (Bottom 3m line)
            y_pos = center_y + attack_line_px
            if point_idx == 0: # Left
                target_x, target_y = self.margin_x, y_pos
            elif point_idx == 1: # Right
                target_x, target_y = self.img_width - self.margin_x, y_pos
                
        elif phase_idx == 2: # Net (Center line)
            y_pos = center_y
            if point_idx == 0: # Left
                target_x, target_y = self.margin_x, y_pos
            elif point_idx == 1: # Right
                target_x, target_y = self.img_width - self.margin_x, y_pos
                
        elif phase_idx == 3: # Far 3m (Top 3m line)
            y_pos = center_y - attack_line_px
            if point_idx == 0: # Left
                target_x, target_y = self.margin_x, y_pos
            elif point_idx == 1: # Right
                target_x, target_y = self.img_width - self.margin_x, y_pos

        # Draw the visual cue if coordinates are valid
        if target_x != -1 and target_y != -1:
            # Draw a target marker (Yellow circle with red border)
            cv2.circle(img, (target_x, target_y), 15, (0, 0, 255), 2) # Red ring
            cv2.circle(img, (target_x, target_y), 8, (0, 255, 255), -1) # Yellow fill
            
            # Optional: Add text label
            cv2.putText(img, "CLICK HERE", (target_x + 20, target_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            cv2.putText(img, "CLICK HERE", (target_x + 20, target_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
        return img

    def get_warped_frame(self, frame, points):
        """
        Generates a synthetic radar view of the court.
        Calculates the homography matrix M for future tracking projections.
        """
        if not points or len(points) < 4:
            return None

        # 1. Calculate Homography Matrix (needed for future tracking)
        self.update_homography(points)
        
        # 2. Create Synthetic Background
        radar_img = self._draw_static_court()
        
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

    def update_player_positions(self, radar_img, tracks):
        """
        Projects tracked players onto the radar view using the homography matrix.
        """
        if self.M is None:
            return radar_img

        # Prepare points for transformation
        # We need a list of (x, y) coordinates representing the feet of each player
        points_to_transform = []
        track_ids = []

        for track in tracks:
            if not track.is_confirmed():
                continue
            
            ltrb = track.to_ltrb() # left, top, right, bottom
            x1, y1, x2, y2 = ltrb
            
            # Estimate feet position: bottom center of the bounding box
            feet_x = (x1 + x2) / 2
            feet_y = y2 
            
            points_to_transform.append([feet_x, feet_y])
            track_ids.append(track.track_id)

        if not points_to_transform:
            # Even if no players, draw the buttons
            self.draw_buttons(radar_img)
            return radar_img

        # Format for cv2.perspectiveTransform: (N, 1, 2)
        src_pts_players = np.array(points_to_transform, dtype="float32").reshape(-1, 1, 2)
        
        # Apply Homography
        dst_pts_players = cv2.perspectiveTransform(src_pts_players, self.M)

        # Draw points on radar
        for i, pt in enumerate(dst_pts_players):
            x, y = int(pt[0][0]), int(pt[0][1])
            
            # Apply Inversion if active (Rotate 180 degrees around center)
            if self.invert_sides:
                x = self.img_width - x
                y = self.img_height - y
                
            # Apply Mirror LR if active
            if self.mirror_lr:
                x = self.img_width - x
            
            tid = track_ids[i]
            
            # Draw player ONLY if projected coordinates are within radar image bounds
            if (0 <= x < self.img_width and 0 <= y < self.img_height):
                # Draw Player Position (Red Circle)
                cv2.circle(radar_img, (x, y), 8, (0, 0, 255), -1)
                
                # Draw Player ID
                # Put text slightly above the dot
                cv2.putText(radar_img, str(tid), (x - 5, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            else:
                print(f"WARNING: Player {tid} projected outside radar image bounds! Original: ({points_to_transform[i][0]}, {points_to_transform[i][1]})")
        
        # Draw Interface Elements (Buttons)
        self.draw_buttons(radar_img)

        return radar_img