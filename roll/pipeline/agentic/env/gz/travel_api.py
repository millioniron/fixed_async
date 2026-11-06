"""Travel API module for flight booking and management services."""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union


class TravelApi:
    """A class representing a travel and flight booking platform."""

    def __init__(self):
        self.wifi: bool = False
        self.logged_in: bool = True

        self.users: Dict[str, Dict[str, Any]] = {
            "user1": {
                "user_name": "Eve",
                "password": "password123",
                "cash_balance": 2000.0,
                "bank_balance": 50000.0,
                "membership_level": "regular",
            },
            "user2": {
                "user_name": "Frank",
                "password": "password456",
                "cash_balance": 8000.0,
                "bank_balance": 8000.0,
                "membership_level": "silver",
            },
            "user3": {
                "user_name": "Grace",
                "password": "password789",
                "cash_balance": 1000.0,
                "bank_balance": 5000.0,
                "membership_level": "gold",
            },
        }

        self.flights: List[Dict[str, Any]] = [
            {
                "flight_no": "CA1234",
                "origin": "Beijing",
                "destination": "Shanghai",
                "depart_time": "2024-07-15 08:00:00",
                "arrival_time": "2024-07-15 10:30:00",
                "status": "available",
                "seats_available": 5,
                "economy_price": 1200,
                "business_price": 3000,
            },
            {
                "flight_no": "MU5678",
                "origin": "Shanghai",
                "destination": "Beijing",
                "depart_time": "2024-07-16 09:00:00",
                "arrival_time": "2024-07-16 11:30:00",
                "status": "available",
                "seats_available": 3,
                "economy_price": 1900,
                "business_price": 3000,
            },
            {
                "flight_no": "CZ4321",
                "origin": "Shanghai",
                "destination": "Beijing",
                "depart_time": "2024-07-16 20:00:00",
                "arrival_time": "2024-07-16 22:00:00",
                "status": "available",
                "seats_available": 8,
                "economy_price": 2500,
                "business_price": 4000,
            },
            {
                "flight_no": "CZ4352",
                "origin": "Shanghai",
                "destination": "Beijing",
                "depart_time": "2024-07-17 20:00:00",
                "arrival_time": "2024-07-17 22:00:00",
                "status": "available",
                "seats_available": 8,
                "economy_price": 1600,
                "business_price": 2500,
            },
            {
                "flight_no": "MU3561",
                "origin": "Beijing",
                "destination": "Nanjing",
                "depart_time": "2024-07-18 08:00:00",
                "arrival_time": "2024-07-18 10:00:00",
                "status": "available",
                "seats_available": 8,
                "economy_price": 1500,
                "business_price": 4000,
            },
            {
                "flight_no": "MU1566",
                "origin": "Beijing",
                "destination": "Nanjing",
                "depart_time": "2024-07-18 20:00:00",
                "arrival_time": "2024-07-18 22:00:00",
                "status": "available",
                "seats_available": 8,
                "economy_price": 1500,
                "business_price": 4000,
            },
            {
                "flight_no": "CZ1765",
                "origin": "Nanjing",
                "destination": "Shenzhen",
                "depart_time": "2024-07-17 20:30:00",
                "arrival_time": "2024-07-17 22:00:00",
                "status": "available",
                "seats_available": 8,
                "economy_price": 1500,
                "business_price": 2500,
            },
            {
                "flight_no": "CZ1765",
                "origin": "Nanjing",
                "destination": "Shenzhen",
                "depart_time": "2024-07-18 12:30:00",
                "arrival_time": "2024-07-18 15:00:00",
                "status": "available",
                "seats_available": 8,
                "economy_price": 1500,
                "business_price": 2500,
            },
            {
                "flight_no": "MH1765",
                "origin": "Xiamen",
                "destination": "Chengdu",
                "depart_time": "2024-07-17 12:30:00",
                "arrival_time": "2024-07-17 15:00:00",
                "status": "available",
                "seats_available": 8,
                "economy_price": 1500,
                "business_price": 2500,
            },
            {
                "flight_no": "MH2616",
                "origin": "Chengdu",
                "destination": "Xiamen",
                "depart_time": "2024-07-18 18:30:00",
                "arrival_time": "2024-07-18 21:00:00",
                "status": "available",
                "seats_available": 8,
                "economy_price": 1500,
                "business_price": 2500,
            },
            {
                "flight_no": "MH2616",
                "origin": "Chengdu",
                "destination": "Fuzhou",
                "depart_time": "2024-07-16 18:30:00",
                "arrival_time": "2024-07-16 21:00:00",
                "status": "available",
                "seats_available": 8,
                "economy_price": 1500,
                "business_price": 2500,
            },
        ]

        self.reservations: List[Dict[str, Any]] = [
            {
                "reservation_id": "res_1",
                "user_id": "user1",
                "flight_no": "CA1234",
                "payment_method": "bank",
                "cabin": "Economy Class",
                "baggage": 1,
                "origin": "Beijing",
                "destination": "Shanghai",
            },
            {
                "reservation_id": "res_2",
                "user_id": "user1",
                "flight_no": "MU5678",
                "payment_method": "bank",
                "cabin": "Business Class",
                "baggage": 1,
                "origin": "Shanghai",
                "destination": "Beijing",
            },
            {
                "reservation_id": "res_3",
                "user_id": "user2",
                "flight_no": "MH1765",
                "payment_method": "bank",
                "cabin": "Business Class",
                "baggage": 1,
                "origin": "Xiamen",
                "destination": "Chengdu",
            },
            {
                "reservation_id": "res_4",
                "user_id": "user2",
                "flight_no": "MU2616",
                "payment_method": "bank",
                "cabin": "Business Class",
                "baggage": 1,
                "origin": "Chengdu",
                "destination": "Xiamen",
            },
        ]

    def load_config(self, config: dict) -> None:
        """Load initial configuration."""
        self.wifi = config.get("wifi", False)
        self.logged_in = config.get("logged_in", True)
        if "users" in config:
            self.users = config["users"]
        if "flights" in config:
            self.flights = config["flights"]
        if "reservations" in config:
            self.reservations = config["reservations"]

    def get_state(self) -> dict:
        """Get current state for comparison."""
        return {
            "users": {k: dict(v) for k, v in self.users.items()},
            "flights": [dict(f) for f in self.flights],
            "reservations": [dict(r) for r in self.reservations],
        }

    # ============ Private Helpers ============

    _BAGGAGE_ALLOWANCE = {
        "regular": {"Economy Class": 1, "Business Class": 2},
        "silver": {"Economy Class": 2, "Business Class": 3},
        "gold": {"Economy Class": 3, "Business Class": 3},
    }

    def _authenticate_user(self, user_id: str, password: str) -> Optional[Dict]:
        """Authenticate user and return user dict, or None on failure."""
        user = self.users.get(user_id)
        if user and user["password"] == password:
            return user
        return None

    def _calculate_baggage_fee(self, membership_level: str, cabin_class: str, baggage_count: int) -> int:
        """Calculate extra baggage fee beyond free allowance."""
        free_limit = self._BAGGAGE_ALLOWANCE.get(membership_level, {}).get(cabin_class, 0)
        return max(baggage_count - free_limit, 0) * 50

    def _update_balance(self, user: Dict, payment_method: str, amount: float) -> bool:
        """Update user balance. Returns False if insufficient funds."""
        key = "cash_balance" if payment_method == "cash" else "bank_balance"
        if user[key] + amount < 0:
            return False
        user[key] += amount
        return True

    def _calculate_price_difference(self, flight: Dict, old_cabin: str, new_cabin: str) -> float:
        """Calculate price difference between cabin classes."""
        prices = {"Economy Class": flight["economy_price"], "Business Class": flight["business_price"]}
        return prices.get(new_cabin, 0) - prices.get(old_cabin, 0)

    # ============ Tool Functions ============

    def get_flight_details(self, origin: str = None, destination: str = None) -> Union[List[Dict], str]:
        """Query flight information based on origin and destination.

        Args:
            origin: The origin city of the flight.
            destination: The destination city of the flight.
        """
        flights = self.flights
        if origin:
            flights = [f for f in flights if f["origin"] == origin]
        if destination:
            flights = [f for f in flights if f["destination"] == destination]

        if not flights:
            return "There are no direct flights that meet the criteria."

        return [
            {
                "flight_no": f["flight_no"],
                "origin": f["origin"],
                "destination": f["destination"],
                "depart_time": f["depart_time"],
                "arrival_time": f["arrival_time"],
                "status": f["status"],
                "seats_available": f["seats_available"],
                "economy_price": f["economy_price"],
                "business_price": f["business_price"],
            }
            for f in flights
        ]

    def find_transfer_flights(
        self, origin_city: str, transfer_city: str, destination_city: str
    ) -> Union[List[Dict], str]:
        """Find connecting flights from origin to destination via a transfer city.

        Args:
            origin_city: The starting city for the flight.
            transfer_city: The city where the transfer takes place.
            destination_city: The final destination city.
        """
        first_leg_flights = [
            f for f in self.flights
            if f["origin"] == origin_city and f["destination"] == transfer_city and f["status"] == "available"
        ]
        second_leg_flights = [
            f for f in self.flights
            if f["origin"] == transfer_city and f["destination"] == destination_city and f["status"] == "available"
        ]

        transfer_flights = []
        for first in first_leg_flights:
            first_arrival = datetime.strptime(first["arrival_time"], "%Y-%m-%d %H:%M:%S")
            for second in second_leg_flights:
                second_departure = datetime.strptime(second["depart_time"], "%Y-%m-%d %H:%M:%S")
                if first_arrival < second_departure:
                    transfer_flights.append({"first_leg": first, "second_leg": second})

        if transfer_flights:
            return transfer_flights
        return "No connecting flights that meet the criteria were found."

    def reserve_flight(
        self,
        user_id: str,
        password: str,
        flight_no: str,
        cabin: str,
        payment_method: str,
        baggage_count: int,
    ) -> str:
        """Reserve a flight for a user.

        Args:
            user_id: The ID of the user making the reservation.
            password: The password of the user for authentication.
            flight_no: The flight number to book.
            cabin: Cabin class, either 'Economy Class' or 'Business Class'.
            payment_method: Payment method, either 'cash' or 'bank'.
            baggage_count: Number of checked baggage items.
        """
        user = self._authenticate_user(user_id, password)
        if not user:
            return "Authentication failed. Please check your user ID and password."

        flight = next((f for f in self.flights if f["flight_no"] == flight_no and f["status"] == "available"), None)
        if not flight:
            return "Flight not found or not available."

        price = flight["economy_price"] if cabin == "Economy Class" else flight["business_price"]
        baggage_fee = self._calculate_baggage_fee(user["membership_level"], cabin, baggage_count)
        total_cost = price + baggage_fee

        if payment_method not in ("cash", "bank"):
            return "Invalid payment method"

        balance_key = "cash_balance" if payment_method == "cash" else "bank_balance"
        if total_cost > self.users[user_id][balance_key]:
            return f"Your {payment_method} balance is insufficient. Please consider using another payment method."

        self.users[user_id][balance_key] -= total_cost
        flight["seats_available"] -= 1

        reservation_id = f"res_{len(self.reservations) + 1}"
        self.reservations.append({
            "reservation_id": reservation_id,
            "user_id": user_id,
            "flight_no": flight_no,
            "payment_method": payment_method,
            "cabin": cabin,
            "baggage": baggage_count,
        })

        return f"Booking successful. Reservation ID: {reservation_id}. Total cost: {total_cost} yuan (including baggage fees)."

    def get_reservation_details(
        self, reservation_id: str = None, user_id: str = None
    ) -> Union[List[Dict], Dict[str, str]]:
        """Query reservation information based on reservation ID or user ID.

        Args:
            reservation_id: The reservation ID of the booking.
            user_id: The ID of the user.
        """
        if reservation_id:
            reservations = [r for r in self.reservations if r["reservation_id"] == reservation_id]
        elif user_id:
            reservations = [r for r in self.reservations if r["user_id"] == user_id]
        else:
            return {"status": "error", "message": "Please provide a valid reservation ID or user ID"}

        detailed = []
        for res in reservations:
            flight_info = next((f for f in self.flights if f["flight_no"] == res["flight_no"]), None)
            detailed.append({**res, "flight_info": flight_info})
        return detailed

    def modify_flight(
        self,
        user_id: str,
        reservation_id: str,
        new_flight_no: str = None,
        new_cabin: str = None,
        add_baggage: int = 0,
        new_payment_method: str = None,
    ) -> str:
        """Modify an existing flight reservation.

        Args:
            user_id: The ID of the user making the modification.
            reservation_id: The reservation ID to modify.
            new_flight_no: New flight number to change to.
            new_cabin: New cabin class, either 'Economy Class' or 'Business Class'.
            add_baggage: Number of additional baggage items to add.
            new_payment_method: Payment method for the modification, 'cash' or 'bank'.
        """
        reservation = next(
            (r for r in self.reservations if r["reservation_id"] == reservation_id and r["user_id"] == user_id),
            None,
        )
        if not reservation:
            return "Reservation not found or user ID does not match."

        current_flight = next((f for f in self.flights if f["flight_no"] == reservation["flight_no"]), None)
        if not current_flight:
            return "Flight information not found."

        payment_method = new_payment_method if new_payment_method else reservation["payment_method"]
        user = self.users.get(user_id)
        if not user:
            return "User information not found."

        result_messages = []

        # Update flight number
        if new_flight_no and new_flight_no != reservation["flight_no"]:
            new_flight = next((f for f in self.flights if f["flight_no"] == new_flight_no), None)
            if new_flight and new_flight["origin"] == current_flight["origin"] \
                    and new_flight["destination"] == current_flight["destination"]:
                reservation["flight_no"] = new_flight_no
                result_messages.append("Flight number has been changed.")
            else:
                return "Flight change failed: Invalid new flight number or destination does not match."

        # Update cabin
        if new_cabin and new_cabin != reservation.get("cabin"):
            price_diff = self._calculate_price_difference(current_flight, reservation["cabin"], new_cabin)
            reservation["cabin"] = new_cabin
            if price_diff > 0:
                if self._update_balance(user, payment_method, -price_diff):
                    result_messages.append(f"Cabin change successful. Price difference paid: {price_diff}.")
                else:
                    result_messages.append("Insufficient balance to pay the cabin price difference.")
            elif price_diff < 0:
                self._update_balance(user, payment_method, -price_diff)
                result_messages.append(f"Cabin change successful. Price difference refunded: {-price_diff}.")

        # Add baggage
        if add_baggage > 0:
            max_free = self._BAGGAGE_ALLOWANCE.get(user["membership_level"], {}).get(reservation["cabin"], 0)
            current_baggage = reservation.get("baggage", 0)
            total_baggage = current_baggage + add_baggage
            extra = max(0, total_baggage - max_free)
            cost = extra * 50
            if cost > 0:
                if self._update_balance(user, payment_method, -cost):
                    result_messages.append(f"Baggage has been added. Additional fee to be paid: {cost}.")
                else:
                    result_messages.append("Insufficient balance to pay the additional baggage fees.")
            reservation["baggage"] = total_baggage

        if not result_messages:
            result_messages.append("Modification completed with no additional fees.")
        return " ".join(result_messages)

    def cancel_reservation(self, user_id: str, reservation_id: str, reason: str) -> str:
        """Cancel a flight reservation with optional refund.

        Args:
            user_id: The ID of the user requesting cancellation.
            reservation_id: The reservation ID to cancel.
            reason: The reason for cancellation.
        """
        current_time = datetime(2024, 7, 14, 6, 0, 0)

        user = self.users.get(user_id)
        if not user:
            return "Invalid user ID."

        reservation = next(
            (r for r in self.reservations if r["reservation_id"] == reservation_id and r["user_id"] == user_id),
            None,
        )
        if not reservation:
            return "Invalid reservation ID or it does not belong to the user."

        flight = next((f for f in self.flights if f["flight_no"] == reservation["flight_no"]), None)
        if not flight:
            return "Invalid flight information."

        depart_time = datetime.strptime(flight["depart_time"], "%Y-%m-%d %H:%M:%S")
        if current_time > depart_time:
            return "The flight segment has been used and cannot be canceled."

        flight_price = flight["economy_price"] if reservation["cabin"] == "Economy Class" else flight["business_price"]
        time_until = depart_time - current_time

        if reason == "The airline has canceled the flight.":
            user["cash_balance"] += flight_price
            return (
                f"The flight has been canceled. Your reservation will be canceled free of charge, "
                f"and {flight_price} yuan has been refunded."
            )

        from datetime import timedelta
        if time_until > timedelta(days=1):
            user["cash_balance"] += flight_price
            return f"More than 24 hours before departure. Free cancellation successful, {flight_price} yuan has been refunded."

        cancel_fee = flight_price * 0.1
        refund = flight_price - cancel_fee
        user["cash_balance"] += refund
        return (
            f"Less than 24 hours before departure. A cancellation fee of {cancel_fee} yuan "
            f"has been deducted, and {refund} yuan has been refunded."
        )

    def get_user_details(self, user_id: str, password: str) -> Union[Dict, Dict[str, str]]:
        """Query user information based on user ID and password.

        Args:
            user_id: The ID of the user to query.
            password: The password of the user for authentication.
        """
        user = self.users.get(user_id)
        if user and user["password"] == password:
            return {k: v for k, v in user.items() if k != "password"}
        return {"status": "error", "message": "Incorrect username or password."}

    def get_baggage_allowance(self, membership_level: str, cabin_class: str) -> int:
        """Get free checked baggage allowance based on membership level and cabin class.

        Args:
            membership_level: Membership level ('regular', 'silver', or 'gold').
            cabin_class: Cabin class ('Economy Class' or 'Business Class').
        """
        return self._BAGGAGE_ALLOWANCE.get(membership_level, {}).get(cabin_class, 0)
