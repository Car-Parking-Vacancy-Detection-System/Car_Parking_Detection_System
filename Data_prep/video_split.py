import cv2
from pathlib import Path
import subprocess
import logging
from datetime import datetime, timedelta
import os

class VideoSplitter:
    def __init__(self, output_dir: str, segment_length: int = 1800):
        """
        Initialize VideoSplitter
        
        Args:
            output_dir (str): Directory to save split videos
            segment_length (int): Length of each segment in seconds (default: 300 seconds = 5 minutes)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.segment_length = segment_length
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def split_video(self, video_path: str) -> list:
        """
        Split a video into smaller segments using ffmpeg
        """
        video_path = Path(video_path)
        video_name = video_path.stem
        
        # Create folder for this video's segments
        segments_dir = self.output_dir / video_name
        segments_dir.mkdir(exist_ok=True)
        
        # Get video duration using ffmpeg
        duration_cmd = [
            'ffmpeg', '-i', str(video_path), 
            '2>&1', '|', 'grep', 'Duration', 
            '|', 'cut', '-d', ' ', '-f', '4', 
            '|', 'sed', "s/,//"
        ]
        
        try:
            # Use ffmpeg to split the video
            command = [
                'ffmpeg',
                '-i', str(video_path),
                '-c', 'copy',  # Copy without re-encoding
                '-f', 'segment',
                '-segment_time', str(self.segment_length),
                '-reset_timestamps', '1',
                f'{segments_dir}/{video_name}_part_%03d.mp4'
            ]
            
            self.logger.info(f"Splitting video: {video_path.name}")
            subprocess.run(command, check=True)
            
            # Get list of generated segments
            segments = sorted(segments_dir.glob('*.mp4'))
            self.logger.info(f"Successfully split {video_path.name} into {len(segments)} segments")
            
            return [str(seg) for seg in segments]
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error splitting {video_path.name}: {str(e)}")
            return []
    
    def split_multiple_videos(self, video_dir: str) -> dict:
        """
        Split multiple videos in a directory
        """
        video_dir = Path(video_dir)
        video_paths = list(video_dir.glob('*.mp4'))  # Add more extensions if needed
        results = {}
        
        self.logger.info(f"Found {len(video_paths)} videos to split")
        
        for video_path in video_paths:
            segments = self.split_video(str(video_path))
            results[video_path.name] = {
                'num_segments': len(segments),
                'segments': segments
            }
        
        # Save summary to log file
        summary_path = self.output_dir / 'splitting_summary.txt'
        with open(summary_path, 'w') as f:
            f.write("Video Splitting Summary\n")
            f.write("=====================\n\n")
            for video, result in results.items():
                f.write(f"Video: {video}\n")
                f.write(f"Number of segments: {result['num_segments']}\n")
                f.write("Segments:\n")
                for seg in result['segments']:
                    f.write(f"  - {Path(seg).name}\n")
                f.write("-" * 30 + "\n")
        
        return results

def main():
    # Configure these paths
    VIDEO_DIR = "Converted"
    OUTPUT_DIR = "split_videos"
    
    # Create splitter (30 minute segments)
    splitter = VideoSplitter(OUTPUT_DIR, segment_length=1800)
    
    # Split all videos
    results = splitter.split_multiple_videos(VIDEO_DIR)
    
    print("\nSplitting complete!")
    print(f"Check {OUTPUT_DIR}/splitting_summary.txt for details")

if __name__ == "__main__":
    main()