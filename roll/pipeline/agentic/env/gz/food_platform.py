"""Food Platform API module for food delivery services."""

from typing import Dict, List, Union


class FoodPlatform:
    """A class representing a food delivery platform."""

    def __init__(self):
        self.wifi = False
        self.logged_in = True

        self.users: Dict[str, Dict[str, Union[str, float]]] = {
            "Eve": {"user_id": "U100", "password": "password123", "balance": 500.0},
            "Frank": {"user_id": "U101", "password": "password456", "balance": 300.0},
            "Grace": {"user_id": "U102", "password": "password789", "balance": 150.0},
            "Helen": {"user_id": "U103", "password": "password321", "balance": 800.0},
            "Isaac": {"user_id": "U104", "password": "password654", "balance": 400.0},
            "Jack": {"user_id": "U105", "password": "password654", "balance": 120.0},
        }

        self.merchant_list: Dict[str, Dict] = {
            "Domino's": {
                "merchant_id": "M100",
                "service_type": "Pizza",
                "menu": [
                    {"product": "Margherita Pizza", "price": 68.0},
                    {"product": "Super Supreme Pizza", "price": 88.0},
                ],
            },
            "Rice Village Bibimbap": {
                "merchant_id": "M101",
                "service_type": "Bibimbap",
                "menu": [
                    {"product": "Stone Pot Bibimbap", "price": 35.0},
                    {"product": "Korean Beef Bibimbap", "price": 45.0},
                ],
            },
            "Haidilao": {
                "merchant_id": "M102",
                "service_type": "Hotpot",
                "menu": [
                    {"product": "Beef Rolls", "price": 68.0},
                    {"product": "Seafood Platter", "price": 88.0},
                ],
            },
            "Heytea": {
                "merchant_id": "M103",
                "service_type": "Milk Tea",
                "menu": [
                    {"product": "Cheese Milk Tea", "price": 25.0},
                    {"product": "Four Seasons Spring Milk Tea", "price": 22.0},
                ],
            },
            "Hema Fresh": {
                "merchant_id": "M104",
                "service_type": "Fresh Grocery",
                "menu": [
                    {"product": "Organic Vegetable Pack", "price": 15.0},
                    {"product": "Fresh Gift Pack", "price": 99.0},
                ],
            },
            "Jiutian BBQ": {
                "merchant_id": "M105",
                "service_type": "BBQ",
                "menu": [
                    {"product": "Korean Grilled Beef", "price": 128.0},
                    {"product": "Grilled Pork Belly", "price": 78.0},
                ],
            },
        }

        self.logged_in_users: List[str] = []
        self.orders: List[Dict] = []

    def load_config(self, config: dict) -> None:
        """Load initial configuration."""
        self.wifi = config.get("wifi", False)
        self.logged_in = config.get("logged_in", True)
        self.logged_in_users = config.get("logged_in_users", [])

    def get_state(self) -> dict:
        """Get current state for comparison."""
        return {
            "users": {k: dict(v) for k, v in self.users.items()},
            "logged_in_users": list(self.logged_in_users),
            "orders": list(self.orders)
        }

    def login_food_platform(self, username: str, password: str) -> Dict[str, Union[bool, str]]:
        """
        Log in to the food platform.

        Args:
            username (str): The username of the user.
            password (str): The password of the user.
        """
        if not self.wifi:
            return {"status": False, "message": "Wi-Fi is not enabled, unable to login"}
        if username not in self.users:
            return {"status": False, "message": "User does not exist"}
        if self.users[username]["password"] != password:
            return {"status": False, "message": "Incorrect password"}
        if username in self.logged_in_users:
            return {"status": False, "message": f"{username} is already logged in"}

        self.logged_in_users.append(username)
        return {"status": True, "message": f"User {username} has successfully logged in!"}

    def get_products(self, merchant_name: str) -> Union[List[Dict], Dict[str, Union[bool, str]]]:
        """
        Retrieve the product list of a specific merchant.

        Args:
            merchant_name (str): Name of the merchant.
        """
        merchant = self.merchant_list.get(merchant_name)
        if merchant:
            return merchant["menu"]
        return {"status": False, "message": f"Merchant '{merchant_name}' does not exist"}

    def add_food_delivery_order(
        self,
        username: str,
        merchant_name: str,
        items: List[Dict[str, Union[str, int]]]
    ) -> Dict[str, Union[bool, str]]:
        """
        Place a food delivery order.

        Args:
            username (str): The username placing the order.
            merchant_name (str): The name of the merchant.
            items (list): List of items to order, each with product name and quantity.
        """
        if username not in self.logged_in_users:
            return {"status": False, "message": f"User {username} is not logged in to the food platform"}
        if merchant_name not in self.merchant_list:
            return {"status": False, "message": "Merchant does not exist"}

        total_price = 0.0
        order_items = []

        for item in items:
            product_name = item.get("product")
            quantity = item.get("quantity", 1)

            if not isinstance(quantity, int) or quantity <= 0:
                return {"status": False, "message": f"Invalid quantity {quantity} for product {product_name}"}

            product_found = False
            for product in self.merchant_list[merchant_name]["menu"]:
                if product["product"] == product_name:
                    total_price += product["price"] * quantity
                    order_items.append({
                        "product": product_name,
                        "quantity": quantity,
                        "price_per_unit": product["price"]
                    })
                    product_found = True
                    break

            if not product_found:
                return {"status": False, "message": f"Product {product_name} does not exist in {merchant_name}'s menu"}

        if total_price > self.users[username]["balance"]:
            return {"status": False, "message": "Insufficient balance to place the order"}

        self.users[username]["balance"] -= total_price
        order = {
            "user_name": username,
            "merchant_name": merchant_name,
            "items": order_items,
            "total_price": total_price,
        }
        self.orders.append(order)
        return {"status": True, "message": f"Food delivery order successfully placed with {merchant_name}. Total amount: {total_price} yuan"}

    def check_balance(self, user_name: str) -> Union[float, Dict[str, Union[bool, str]]]:
        """
        Check the balance of the specified user.

        Args:
            user_name (str): The username to check balance for.
        """
        if user_name in self.users:
            return self.users[user_name]["balance"]
        return {"status": False, "message": f"User {user_name} does not exist!"}

    def view_orders(self, user_name: str) -> Dict[str, Union[bool, str, List]]:
        """
        View all orders of a user.

        Args:
            user_name (str): The username.
        """
        user_orders = [order for order in self.orders if order["user_name"] == user_name]
        if not user_orders:
            return {"status": False, "message": "User has no order records"}
        return {"status": True, "orders": user_orders}
