"""Base API module for device-level operations like WiFi and login."""

from typing import Dict, Union


class BaseApi:
    """Base API for device-level operations."""

    def __init__(self):
        self.wifi = False
        self.logged_in = True

    def load_config(self, config: dict) -> None:
        """Load initial configuration."""
        self.wifi = config.get("wifi", False)
        self.logged_in = config.get("logged_in", True)

    def get_state(self) -> dict:
        """Get current state for comparison."""
        return {
            "wifi": self.wifi,
            "logged_in": self.logged_in
        }

    def turn_on_wifi(self) -> Dict[str, Union[bool, str]]:
        """
        Turn on WiFi connection.

        Returns:
            Dict containing status and message.
        """
        self.wifi = True
        return {"status": True, "message": "Wi-Fi has been turned on"}

    def turn_off_wifi(self) -> Dict[str, Union[bool, str]]:
        """
        Turn off WiFi connection.

        Returns:
            Dict containing status and message.
        """
        self.wifi = False
        return {"status": True, "message": "Wi-Fi has been turned off"}

    def login_device(self) -> Dict[str, Union[bool, str]]:
        """
        Log in to the device.

        Returns:
            Dict containing status and message.
        """
        self.logged_in = True
        return {"status": True, "message": "Device has been logged in"}

    def logout_device(self) -> Dict[str, Union[bool, str]]:
        """
        Log out from the device.

        Returns:
            Dict containing status and message.
        """
        self.logged_in = False
        return {"status": True, "message": "Device has been logged out"}
