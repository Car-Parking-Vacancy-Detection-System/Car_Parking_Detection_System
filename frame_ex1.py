import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import concurrent.futures
import logging
from typing import List, Set

class OrganizedFrameExtractor:
    def __init__(self, base_output_dir: str):
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.logger = logging.getLogger(__name__)
        
        # Parameters for frame extraction
        self.min_frame_difference = 45.0  # Increased threshold for more distinct frames
        self.frame_interval = 15  # Check frames more frequently
        self.blur_threshold = 100  # Threshold for blurry frame detection

    def _create_video_folder(self, original_video_name: str) -> Path:
        """Create a folder for each original video's frames"""
        # Clean up the video name to remove any problematic characters
        safe_name = "".join(c for c in original_video_name if c.isalnum() or c in (' ', '-', '_'))
        folder_path = self.base_output_dir / safe_name
        folder_path.mkdir(exist_ok=True)
        return folder_path
        
    def _calculate_frame_difference(self, frame: np.ndarray, prev_frame: np.ndarray) -> float:
        """Calculate difference between frames using multiple metrics"""
        if prev_frame is None:
            return float('inf')
            
        # Convert to grayscale
        gray1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate absolute difference
        diff = cv2.absdiff(gray1, gray2)
        mean_diff = np.mean(diff)
        
        # Calculate structural similarity (lower means more different)
        ssim = cv2.matchTemplate(gray1, gray2, cv2.TM_CCOEFF_NORMED)[0][0]
        
        # Combine metrics (you can adjust weights)
        combined_diff = mean_diff * (1 - ssim)
        
        return combined_diff
        
    def _is_frame_blurry(self, frame: np.ndarray) -> bool:
        """Detect if frame is too blurry"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var < self.blur_threshold
        
    def _is_frame_distinct(self, frame: np.ndarray, prev_frame: np.ndarray) -> bool:
        """Check if frame is distinct enough to save"""
        if prev_frame is None:
            return not self._is_frame_blurry(frame)
            
        # Skip blurry frames
        if self._is_frame_blurry(frame):
            return False
            
        # Check difference threshold
        diff = self._calculate_frame_difference(frame, prev_frame)
        return diff > self.min_frame_difference
    
    def process_video(self, video_path: str) -> dict:
        """Process single video segment and extract frames"""
        video_path = Path(video_path)
        original_video_name = video_path.stem.rsplit('_part_', 1)[0]
        segment_number = video_path.stem.rsplit('_part_', 1)[1]
        
        output_folder = self._create_video_folder(original_video_name)
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            self.logger.error(f"Error opening video segment: {video_path}")
            return {
                "video": original_video_name,
                "segment": segment_number,
                "frames_saved": 0,
                "status": "error"
            }
        
        frame_count = 0
        saved_count = 0
        prev_frame = None
        
        self.logger.info(f"Processing segment {segment_number} of video: {original_video_name}")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if frame_count % self.frame_interval == 0:
                    if self._is_frame_distinct(frame, prev_frame):
                        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S_%f")
                        filename = f"{original_video_name}_seg{segment_number}_{timestamp}.jpg"
                        output_path = output_folder / filename
                        cv2.imwrite(str(output_path), frame)
                        saved_count += 1
                        prev_frame = frame.copy()
                
                frame_count += 1
                
                if frame_count % 1000 == 0:
                    self.logger.info(f"{original_video_name} - Segment {segment_number}: Processed {frame_count} frames, saved {saved_count}")
                    
        except Exception as e:
            self.logger.error(f"Error processing {original_video_name} segment {segment_number}: {str(e)}")
            return {
                "video": original_video_name,
                "segment": segment_number,
                "frames_saved": saved_count,
                "status": "error"
            }
            
        finally:
            cap.release()
            
        self.logger.info(f"Completed {original_video_name} segment {segment_number}. Saved {saved_count} frames")
        return {
            "video": original_video_name,
            "segment": segment_number,
            "frames_saved": saved_count,
            "status": "success"
        }

    # ... rest of the class methods remain the same ...
    def process_all_videos(self, split_videos_dir: str, max_workers: int = 4) -> List[dict]:
        """Process all video segments in the specified directory structure"""
        split_videos_dir = Path(split_videos_dir)
        # Look for video segments in all subdirectories
        video_paths = []
        for video_dir in split_videos_dir.iterdir():
            if video_dir.is_dir():
                video_paths.extend(video_dir.glob("*.mp4"))
        
        results = []
        
        self.logger.info(f"Found {len(video_paths)} video segments to process")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_video = {executor.submit(self.process_video, str(video_path)): video_path 
                             for video_path in video_paths}
            
            for future in concurrent.futures.as_completed(future_to_video):
                result = future.result()
                results.append(result)
                
        # Print summary
        total_frames = sum(r["frames_saved"] for r in results)
        successful = sum(1 for r in results if r["status"] == "success")
        self.logger.info(f"\nProcessing Summary:")
        self.logger.info(f"Total segments processed: {len(results)}")
        self.logger.info(f"Successfully processed: {successful}")
        self.logger.info(f"Total frames extracted: {total_frames}")
        
        return results

def main():
    # Directory containing your split video segments
    SPLIT_VIDEOS_DIR = "split_videos"
    
    # Base directory where all extracted frames will be saved
    OUTPUT_DIR = "extracted_frames1"
    
    # Create extractor
    extractor = OrganizedFrameExtractor(base_output_dir=OUTPUT_DIR)
    
    # Process all videos
    results = extractor.process_all_videos(SPLIT_VIDEOS_DIR)
    
    # Save processing results to a log file
    log_path = Path(OUTPUT_DIR) / "extraction_summary.txt"
    with open(log_path, "w") as f:
        f.write("Video Processing Results\n")
        f.write("======================\n\n")
        for result in results:
            f.write(f"Video: {result['video']}\n")
            f.write(f"Segment: {result['segment']}\n")
            f.write(f"Frames Saved: {result['frames_saved']}\n")
            f.write(f"Status: {result['status']}\n")
            f.write("-" * 30 + "\n")

if __name__ == "__main__":
    main()