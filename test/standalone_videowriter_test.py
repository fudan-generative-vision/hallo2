import numpy as np
import ffmpeg
import sys

class VideoWriter:
    def __init__(self, video_save_path, height, width, fps, audio):
        if height > 2160:
            print('You are generating video that is larger than 4K, which will be very slow due to IO speed.',
                  'We highly recommend to decrease the outscale(aka, -s).')
        if audio is not None:
            self.stream_writer = (
                ffmpeg.input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'{width}x{height}',
                            framerate=fps).output(
                                audio,
                                video_save_path,
                                pix_fmt='yuv420p',
                                vcodec='libx264',
                                loglevel='error',
                                acodec='copy').overwrite_output().run_async(
                                    pipe_stdin=True, pipe_stdout=True, cmd='ffmpeg'))
        else:
            self.stream_writer = (
                ffmpeg.input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'{width}x{height}',
                            framerate=fps).output(
                                video_save_path, pix_fmt='yuv420p', vcodec='libx264',
                                loglevel='error').overwrite_output().run_async(
                                    pipe_stdin=True, pipe_stdout=True, cmd='ffmpeg'))

    def write_frame(self, frame):
        try:
            frame = frame.astype(np.uint8).tobytes()
            self.stream_writer.stdin.write(frame)
        except BrokenPipeError:
            print('Please re-install ffmpeg and libx264 by running\n',
                  '\t$ conda install -c conda-forge ffmpeg\n',
                  '\t$ conda install -c conda-forge x264')
            sys.exit(0)

    def close(self):
        self.stream_writer.stdin.close()
        self.stream_writer.wait()

def create_test_video(output_path, duration=5, fps=30, width=640, height=480):
    # VideoWriterインスタンスを作成
    writer = VideoWriter(output_path, height, width, fps, audio=None)

    # テスト用の動画データを生成
    total_frames = duration * fps
    
    try:
        for i in range(total_frames):
            # カラフルな動く円を描画
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            cx = int(width/2 + width/4 * np.sin(i*2*np.pi/total_frames))
            cy = int(height/2 + height/4 * np.cos(i*2*np.pi/total_frames))
            color = (
                int(255*np.sin(i*2*np.pi/total_frames)**2),
                int(255*np.cos(i*2*np.pi/total_frames)**2),
                int(255*np.sin(i*4*np.pi/total_frames)**2)
            )
            
            # 円を描画
            xx, yy = np.meshgrid(np.arange(width), np.arange(height))
            circle = ((xx - cx)**2 + (yy - cy)**2) < 50**2
            frame[circle] = color

            # フレームを書き込む
            writer.write_frame(frame)

        # ライターを閉じる
        writer.close()
        print(f"ビデオが正常にエンコードされ、{output_path}に保存されました")

    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")

if __name__ == "__main__":
    output_file = "test_video_standalone.mp4"
    create_test_video(output_file)
    print(f"テスト完了: {output_file}")
