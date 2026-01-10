import glob
import queue
import threading

import sounddevice as sd
import soundfile as sf

from constants import DEFAULT_BLOCKSIZE_SECS

__all__ = ["AudioPlayer"]


class AudioPlayer:
    """Class to play audio files."""

    def __init__(self, msg_q: queue.Queue | None = None):
        """Initialize the player with a worker thread."""
        self.msg_q: queue.Queue | None = msg_q
        self.play_q: queue.Queue = queue.Queue()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop playback and clear the queue."""
        self.stop_event.set()

        # Clear remaining items from the queue
        while not self.play_q.empty():
            try:
                self.play_q.get_nowait()
                self.play_q.task_done()
            except queue.Empty:
                break

    def _worker(self):
        """Worker thread that processes the playback queue."""
        while True:
            # Block until an item is available
            item = self.play_q.get()
            fname, duration = item

            # Check if we should stop before starting playback
            if self.stop_event.is_set():
                self.play_q.task_done()
                continue

            try:
                self._play_file_or_loop(fname, duration)

            except Exception as e:
                self.stop()
                print(f"Error playing {fname}: {e}")

            finally:
                self.play_q.task_done()

    def _play_file_or_loop(self, fname: str, duration: float):
        """Helper to play a single file, possibly looping it.
        Parameters
        ----------
        fname : str
            File name to play.
        duration : float
            Maximum play time in seconds (0 = play once).
        """
        print(f"Playing {fname}", flush=True)
        if self.msg_q is not None:
            self.msg_q.put(fname)

        # Open audio input file
        with sf.SoundFile(fname) as f:
            # Calculate block size
            block_size = int(f.samplerate * DEFAULT_BLOCKSIZE_SECS)

            # Open audio output stream
            with sd.OutputStream(
                samplerate=f.samplerate, channels=f.channels, dtype="float32"
            ) as stream:
                # Read and play audio
                while duration > 0.0:
                    # Return if stop was requested
                    if self.stop_event.is_set():
                        return

                    # Read a block of data
                    data = f.read(block_size)

                    # Handle end of file
                    if len(data) == 0:
                        return

                    # Update duration if greater than zero
                    duration -= len(data) / f.samplerate

                    # Convert block to float32 if necessary
                    if data.dtype != "float32":
                        data = data.astype("float32")

                    # Write data to stream
                    stream.write(data)

    def play(self, pattern: str, duration: float = 0.0, quiet: bool = False) -> None:
        """
        Play audio files matching the given pattern.

        Parameters
        ----------
        pattern : str
            File pattern (expanded with glob.glob) to find audio files.
        duration : float
            Maximum play time in seconds for each file (0 = play files once).
        quiet : bool
            If False, print the audio output device name (default: False).
        """
        files = glob.glob(pattern)
        if not files:
            print("No files found matching the pattern.")
            return

        # Print the output device name if not quiet
        if not quiet:
            device_info = sd.query_devices(sd.default.device[1], "output")
            print(f"Audio output: {device_info['name']}")

        for fname in files:
            try:
                # Push to queue
                self.play_q.put((fname, duration))

                # Block while the thread is busy
                try:
                    self.play_q.join()

                except KeyboardInterrupt:
                    print("\nPlayback interrupted by user.", flush=True)
                    self.stop()
                    break

            except queue.Full:
                print(f"Queue full, skipping {fname}")
                continue

        # Block until all items in the queue have been processed
        self.play_q.join()
