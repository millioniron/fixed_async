"""Reminder API module for managing reminders."""

from typing import Dict, List, Union
from datetime import datetime


class ReminderApi:
    """A class representing a Reminder API for managing reminders."""

    def __init__(self):
        self.wifi = False
        self.logged_in = True
        self.max_capacity = 6

        self.reminder_list: Dict[int, Dict] = {
            1: {
                "reminder_id": 1001,
                "title": "Doctor's Appointment",
                "description": "Visit Dr. Smith for a checkup.",
                "time": "2024-07-15 09:30",
                "notified": False,
            },
            2: {
                "reminder_id": 1002,
                "title": "Team Meeting",
                "description": "Monthly project review with the team.",
                "time": "2024-07-17 11:00",
                "notified": False,
            },
            3: {
                "reminder_id": 1003,
                "title": "To-do list",
                "description": 'First, help Frank place a food delivery order at "Hema Fresh," ordering two "Fresh Gift Packs." Then, send a message to Frank saying, "The price of the purchased goods is () yuan." Replace the parentheses with the actual amount, keeping one decimal place.',
                "time": "2024-07-16 11:00",
                "notified": False,
            },
        }
        self.reminder_id_counter: int = 3

    def load_config(self, config: dict) -> None:
        """Load initial configuration."""
        self.wifi = config.get("wifi", False)
        self.logged_in = config.get("logged_in", True)

    def get_state(self) -> dict:
        """Get current state for comparison."""
        return {"reminder_list": {k: dict(v) for k, v in self.reminder_list.items()}}

    def view_reminder_by_title(self, title: str) -> Dict[str, Union[bool, str, Dict]]:
        """
        View a specific reminder by its title.

        Args:
            title (str): The title of the reminder.
        """
        if not self.logged_in:
            return {"status": False, "message": "The device is not logged in, so you cannot view notifications"}
        for reminder_id, reminder in self.reminder_list.items():
            if reminder["title"] == title:
                return {"status": True, "reminder": reminder}
        return {"status": False, "message": f"No reminder found with the title '{title}'."}

    def add_reminder(self, title: str, description: str, time: str) -> Dict[str, Union[bool, str]]:
        """
        Add a new reminder.

        Args:
            title (str): Reminder title.
            description (str): Reminder description.
            time (str): Reminder time.
        """
        if not self.logged_in:
            return {"status": False, "message": "Device not logged in. Unable to add a new reminder."}
        if len(self.reminder_list) >= self.max_capacity:
            return {"status": False, "message": "Reminder capacity is full. Unable to add a new reminder."}

        self.reminder_id_counter += 1
        reminder_id = self.reminder_id_counter
        self.reminder_list[reminder_id] = {
            "reminder_id": reminder_id,
            "title": title,
            "description": description,
            "time": time,
            "notified": False,
        }
        return {"status": True, "message": f"Reminder '{title}' was successfully added."}

    def delete_reminder(self, reminder_id: int) -> Dict[str, Union[bool, str]]:
        """
        Delete the specified reminder.

        Args:
            reminder_id (int): The ID of the reminder to delete.
        """
        if not self.logged_in:
            return {"status": False, "message": "Device not logged in. Unable to delete the specified reminder."}
        if reminder_id not in self.reminder_list:
            return {"status": False, "message": "Reminder ID does not exist."}

        del self.reminder_list[reminder_id]
        return {"status": True, "message": f"Reminder ID {reminder_id} was successfully deleted."}

    def view_all_reminders(self) -> Dict[str, Union[bool, str, List]]:
        """
        View all reminders.
        """
        if not self.reminder_list:
            return {"status": False, "message": "No reminders found."}

        reminders = []
        for reminder in self.reminder_list.values():
            reminders.append({
                "title": reminder["title"],
                "description": reminder["description"],
                "time": reminder["time"],
                "notified": reminder["notified"],
            })
        return {"status": True, "reminders": reminders}