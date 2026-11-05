#!/usr/bin/env python3
"""
Audio Device Selection CLI

Uses AudioSelect to list and select audio devices.
"""

import argparse
import sys
from audio_select import AudioSelect


def main():
    parser = argparse.ArgumentParser(description="Audio Device Selection CLI")
    parser.add_argument(
        "-l", "--list", action="store_true", help="List devices and exit"
    )
    args = parser.parse_args()

    audio_select = AudioSelect()

    # Sort for better presentation
    audio_select.sort(["category", "name", "api_name"])

    print(audio_select)

    if args.list:
        return

    while True:
        try:
            selection = input("\nEnter device ID to select (or 'q' to quit): ")
            if selection.lower() == "q":
                sys.exit(0)

            device_id = int(selection)

            # Find the device entry matching the user-selected numeric ID.
            # Note: `AudioSelect.sort()` may reorder the list for display, so
            # we must search the entries for the original `id` field assigned
            # when the device list was created (enumeration index).

            selected_device = None
            for dev in audio_select.devices:
                if dev["id"] == device_id:
                    selected_device = dev
                    break

            if selected_device:
                print("\nSelected Device:")
                print(f"  ID:       {selected_device['id']}")
                print(f"  Name:     {selected_device['name']}")
                print(f"  API:      {selected_device['api_name']}")
                print(f"  Category: {selected_device['category']}")
                in_ch = selected_device["max_input_channels"]
                out_ch = selected_device["max_output_channels"]
                print(f"  Channels: {in_ch} in, {out_ch} out")
                print(f"  Sample Rate: {selected_device['default_samplerate']}")
                break
            else:
                print(f"Error: Device ID {device_id} not found.")

        except ValueError:
            print("Invalid input. Please enter a numeric device ID.")
        except KeyboardInterrupt:
            sys.exit(0)


if __name__ == "__main__":
    main()
