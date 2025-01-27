import cv2
import triangulation as tri
import noise_player as nopl

class DepthDisplayer:
    def __init__(self, max_depth, min_depth, warning1, warning2, warning3):
        self.depth_image = None
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.warning1 = warning1
        self.warning2 = warning2
        self.warning3 = warning3
        
        self.noise = nopl.Noise(self.warning1, self.warning2, self.warning3)

    def display_depth(self, circleR, circleL, fR, fL, B, f, alpha, maskR, maskL):
        if circleR is None or circleL is None:
            cv2.putText(fR, "track lost", (75, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(fL, "track lost", (75, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            depth = tri.find_depth(circleR, circleL, fR, fL, B, f, alpha)
            self.noise.setnoise(depth)
            if depth is not None:
                # Base tracking text
                cv2.putText(fR, "tracking", (75, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (124, 252, 0), 2)
                cv2.putText(fL, "tracking", (75, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (124, 252, 0), 2)
                
                # Display depth
                depth_text = f"Distance: {round(depth,3)} cm"
                cv2.putText(fR, depth_text, (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (124, 252, 0), 2)
                cv2.putText(fL, depth_text, (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (124, 252, 0), 2)
                print(f"Depth: {depth} cm")

                # Check depth limits
                if depth > self.max_depth:
                    warning_text = "MAX DEPTH HIT"
                    cv2.putText(fR, warning_text, (75, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(fL, warning_text, (75, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                elif depth < self.min_depth:
                    warning_text = "MIN DEPTH HIT"
                    cv2.putText(fR, warning_text, (75, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(fL, warning_text, (75, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Check warning levels
                if depth <= self.warning1:
                    warning_text = "WARNING 1"
                    cv2.putText(fR, warning_text, (75, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(fL, warning_text, (75, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                elif depth <= self.warning2:
                    warning_text = "WARNING 2"
                    cv2.putText(fR, warning_text, (75, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                    cv2.putText(fL, warning_text, (75, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                    
                elif depth <= self.warning3:
                    warning_text = "WARNING 3"
                    cv2.putText(fR, warning_text, (75, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(fL, warning_text, (75, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            else:
                cv2.putText(fR, "invalid depth", (75, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(fL, "invalid depth", (75, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            
        cv2.imshow("Right Camera", fR)
        cv2.imshow("Left Camera", fL)
        cv2.imshow("Right Mask", maskR)
        cv2.imshow("Left Mask", maskL)