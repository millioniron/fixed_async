from typing import Dict, List, Union
from datetime import datetime


class BaseApi:

    def __init__(self):
        self.wifi = False
        self.logged_in = True

    def _load_scenario(self, scenario: dict, long_context=False) -> None:

        self.wifi = scenario.get("wifi", False)
        self.logged_in = scenario.get("logged_in",True)


    def turn_on_wifi(self):
        self.wifi = True
        return {'status': True, 'message': 'Wi-Fi has been turned on'}

    def login_device(self):
        self.logged_in = True
        return {'status': True, 'message': 'Device has been logged in'}


class MessageFoodPlatform(BaseApi):
    """
    A combined class representing both Message API and Food Platform functionality.
    """

    def __init__(self):
        super().__init__()


        self.available_tools = {
            "turn_on_wifi": self.turn_on_wifi,
            "login_device": self.login_device,
            "send_message": self.send_message,
            "view_messages_between_users": self.view_messages_between_users,
            "delete_message": self.delete_message,
            "search_messages": self.search_messages,
            "get_latest_message_id": self.get_latest_message_id,
            "get_earliest_message_id": self.get_earliest_message_id,
            "login_food_platform": self.login_food_platform,
            "check_balance": self.check_balance,
            "add_food_delivery_order": self.add_food_delivery_order,
            "get_products": self.get_products,
        }




        """
        Initialize the combined platform with users, messages, and food platform features.
        """
        # Initialize six users for messaging
        self.max_capacity = 6
        self.user_list: Dict[str, Dict[str, Union[str, int]]] = {
            "Eve": {
                "user_id": "USR100",
                "phone_number": "123-456-7890",
                "occupation": "Software Engineer",
            },
            "Frank": {
                "user_id": "USR101",
                "phone_number": "234-567-8901",
                "occupation": "Data Scientist",
            },
            "Grace": {
                "user_id": "USR102",
                "phone_number": "345-678-9012",
                "occupation": "Product Manager",
            },
            "Helen": {
                "user_id": "USR103",
                "phone_number": "456-789-0123",
                "occupation": "UX Designer",
            },
            "Isaac": {
                "user_id": "USR104",
                "phone_number": "567-890-1234",
                "occupation": "DevOps Engineer",
            },
            "Jack": {
                "user_id": "USR105",
                "phone_number": "678-901-2345",
                "occupation": "Marketing Specialist",
            },
        }


        # Initialize message history between six users
        # Message 1 pairs with reminder, Message 2 pairs with food
        self.inbox: Dict[int, Dict[str, Union[str, int]]] = {
            1: {
                "sender_id": "USR100",
                "receiver_id": "USR101",
                "message": "Hey Frank, don't forget about our meeting on 2024-06-11 at 4 PM in Conference Room 1.",
                "time": "2024-06-09",
            },
            2: {
                "sender_id": "USR101",
                "receiver_id": "USR102",
                "message": "Can you help me order a \"Margherita Pizza\" delivery? The merchant is Domino's.",
                "time": "2024-03-09",
            },
            3: {
                "sender_id": "USR102",
                "receiver_id": "USR103",
                "message": "Please check the milk tea delivery options available from Heytea and purchase a cheaper milk tea for me. After making the purchase, remember to reply to me with \"Already bought.\"",
                "time": "2023-12-05",
            },
            4: {
                "sender_id": "USR103",
                "receiver_id": "USR102",
                "message": "No problem Helen, I can assist you.",
                "time": "2024-09-09",
            },
            5: {
                "sender_id": "USR104",
                "receiver_id": "USR105",
                "message": "Isaac, are you available for a call?",
                "time": "2024-06-06",
            },
            6: {
                "sender_id": "USR105",
                "receiver_id": "USR104",
                "message": "Yes Jack, let's do it in 30 minutes.",
                "time": "2024-01-15",
            },
        }

        self.message_id_counter: int = 6

        # Initialize users and their initial balances for food platform
        self.users: Dict[str, Dict[str, Union[str, float]]] = {
            "Eve": {"user_id": "U100", "password": "password123", "balance": 500.0},
            "Frank": {"user_id": "U101", "password": "password456", "balance": 300.0},
            "Grace": {"user_id": "U102", "password": "password789", "balance": 150.0},
            "Helen": {"user_id": "U103", "password": "password321", "balance": 800.0},
            "Isaac": {"user_id": "U104", "password": "password654", "balance": 400.0},
            "Jack": {"user_id": "U105", "password": "password654", "balance": 120.0},
        }

        # Initialize six merchants and their menus
        self.merchant_list: Dict[str, Dict[str, Union[str, List[Dict[str, Union[str, float]]]]]] = {
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

        # Initialize list of logged-in users for food platform
        self.logged_in_users: List[str] = []

        # Initialize order history
        self.orders: List[Dict[str, Union[str, int, float, datetime]]] = []


    def _load_scenario(self, scenario: dict, long_context=False) -> None:

        self.wifi = scenario.get("wifi", False)
        self.logged_in = scenario.get("logged_in",True)
        self.logged_in_users = scenario.get("logged_in_users",[])


    # Message API methods
    def send_message(self, sender_name: str, receiver_name: str, message: str) -> Dict[str, Union[bool, str]]:
        """
        Send a message and add it to the message records.
        Args:
            sender_name (str): Sender's username.
            receiver_name (str): Receiver's username.
            message (str): The content of the message to be sent.
        Returns:
            Dict[str, Union[bool, str]]: A dictionary containing the status of the send operation and the result message.
        """
        if self.logged_in == False:
            return {"status": False, "message": "Device not logged in, unable to send message"}

        if self.wifi == False:
            return {"status": False, "message": "Wi-Fi is turned off, cannot send messages at this time"}

        if len(self.inbox) >= self.max_capacity:
            return {"status": False, "message": "Inbox capacity is full. You need to ask the user which message to delete."}

        # Verify that the sender and receiver exist
        if sender_name not in self.user_list or receiver_name not in self.user_list:
            return {"status": False, "message": "Sender or receiver does not exist"}

        sender_id = self.user_list[sender_name]["user_id"]
        receiver_id = self.user_list[receiver_name]["user_id"]

        # Add the message to the inbox
        self.message_id_counter += 1
        self.inbox[self.message_id_counter] = {
            "sender_id": sender_id,
            "receiver_id": receiver_id,
            "message": message,
        }

        return {"status": True, "message": f"Message successfully sent to {receiver_name}."}

    # 2. Delete Message
    def delete_message(self, message_id: int) -> Dict[str, Union[bool, str]]:
        """
        Delete a specified message.
        Args:
            message_id (int): The ID of the message to be deleted.
        Returns:
            Dict[str, Union[bool, str]]: A dictionary containing the status of the delete operation and the result message.
        """
        if self.logged_in == False:
            return {"status": False, "message": "Device not logged in, unable to delete message"}
        if message_id not in self.inbox:
            return {"status": False, "message": "Message ID does not exist"}

        del self.inbox[message_id]
        return {"status": True, "message": f"Message ID {message_id} has been successfully deleted."}

    # 3. View Messages Between Users
    def view_messages_between_users(self, sender_name: str, receiver_name: str) -> Dict[str, Union[bool, str, List[Dict[str, str]]]]:
        """
        View all messages sent from User A to User B.
        Args:
            sender_name (str): Sender's username.
            receiver_name (str): Receiver's username.
        Returns:
            Dict[str, Union[bool, str, List[Dict[str, str]]]]: A dictionary containing the list of messages between the sender and receiver.
        """
        if self.logged_in == False:
            return {"status": False, "message": "Device not logged in, unable to view message information"}

        if sender_name not in self.user_list:
            return {"status": False, "message": "Sender does not exist"}

        if receiver_name not in self.user_list:
            return {"status": False, "message": "Receiver does not exist"}

        sender_id = self.user_list[sender_name]["user_id"]
        receiver_id = self.user_list[receiver_name]["user_id"]
        messages_between_users = []

        # Iterate through the inbox to find messages sent from sender_id to receiver_id
        for msg_id, msg_data in self.inbox.items():
            if msg_data["sender_id"] == sender_id and msg_data["receiver_id"] == receiver_id:
                messages_between_users.append({
                    "id": msg_id,
                    "sender": sender_name,
                    "receiver": receiver_name,
                    "message": msg_data["message"]
                })

        if not messages_between_users:
            return {"status": False, "message": "No related message records found"}

        return {"status": True, "messages": messages_between_users}


    def search_messages(self, user_name: str, keyword: str) -> Dict[str, Union[str, List[Dict[str, str]]]]:
        """
        Search a user's messages based on a keyword, including sent and received messages.
        Args:
            user_name (str): Username.
            keyword (str): Keyword to search.
        Returns:
            Dict[str, Union[str, List[Dict[str, str]]]]: List of matching messages.
        """
        if user_name not in self.user_list:
            return {"status": False, "message": "User does not exist"}

        user_id = self.user_list[user_name]["user_id"]
        matched_messages = []

        # Iterate through the inbox to find messages sent or received by the user that contain the keyword
        for msg_id, msg_data in self.inbox.items():
            if (msg_data["sender_id"] == user_id or msg_data["receiver_id"] == user_id) and keyword.lower() in msg_data["message"].lower():
                matched_messages.append({
                    "id": msg_id,
                    "sender_id": msg_data["sender_id"],
                    "receiver_id": msg_data["receiver_id"],
                    "message": msg_data["message"]
                })

        if not matched_messages:
            return {"status": False, "message": "No messages found containing the keyword"}

        return {"status": True, "messages": matched_messages}

    def get_all_message_times_with_ids(self) -> Union[Dict[int, str], Dict[str, Union[bool, str]]]:
        """
        Get the times of all messages along with their corresponding message IDs.
        Returns:
            Union[Dict[int, str], Dict[str, Union[bool, str]]]: A dictionary with message IDs as keys and times as values, or an error message.
        """
        if self.logged_in == False:
            return {"status": False, "message": "Device not logged in, unable to retrieve all message times and their corresponding message IDs."}
        message_times_with_ids = {msg_id: msg_data["time"] for msg_id, msg_data in self.inbox.items()}
        return message_times_with_ids

    def get_latest_message_id(self) -> Dict[str, Union[bool, str, int]]:
        """
        Get the ID of the latest sent message.
        Returns:
            Dict[str, Union[bool, str, int]]: A dictionary containing the latest message ID.
        """
        if self.logged_in == False:
            return {"status": False, "message": "Device not logged in, unable to retrieve the latest sent message ID."}
        if not self.inbox:
            return {"status": False, "message": "No message records found"}

        # Iterate through all messages to find the latest message based on time
        latest_message_id = None
        latest_time = None

        for message_id, message_data in self.inbox.items():
            message_time = datetime.strptime(message_data['time'], "%Y-%m-%d")
            if latest_time is None or message_time > latest_time:
                latest_time = message_time
                latest_message_id = message_id

        return {"status": True, "message": f"The latest message ID is {latest_message_id}", "message_id": latest_message_id}

    def get_earliest_message_id(self) -> Dict[str, Union[bool, str, int]]:
        """
        Get the ID of the earliest sent message.
        Returns:
            Dict[str, Union[bool, str, int]]: A dictionary containing the earliest message ID.
        """
        if self.logged_in == False:
            return {"status": False, "message": "Device not logged in, unable to retrieve the earliest sent message ID."}
        if not self.inbox:
            return {"status": False, "message": "No message records found"}

        # Iterate through all messages to find the earliest message based on time
        earliest_message_id = None
        earliest_time = None

        for message_id, message_data in self.inbox.items():
            message_time = datetime.strptime(message_data['time'], "%Y-%m-%d")
            if earliest_time is None or message_time < earliest_time:
                earliest_time = message_time
                earliest_message_id = message_id

        return {"status": True, "message": f"The earliest message ID is {earliest_message_id}", "message_id": earliest_message_id}


    # Food Platform methods
    def login_food_platform(self, username: str, password: str) -> Dict[str, Union[bool, str]]:
        if self.wifi == False:
            return {"status": False, "message": "Wi-Fi is not enabled, unable to login"}
        if username not in self.users:
            return {"status": False, "message": "User does not exist"}
        if self.users[username]["password"] != password:
            return {"status": False, "message": "Incorrect password"}

        # Check if the user is already logged in
        if username in self.logged_in_users:
            return {"status": False, "message": f"{username} is already logged in"}

        # Record the logged-in user
        self.logged_in_users.append(username)
        return {"status": True, "message": f"User {username} has successfully logged in!"}


    def view_logged_in_users(self) -> Dict[str, Union[bool, List[str]]]:
        """
        View all currently logged-in users.
        Returns:
            Dict[str, Union[bool, List[str]]]: List of logged-in users.
        """
        if not self.logged_in_users:
            return {"status": False, "message": "No users are currently logged in to the food platform"}

        return {"status": True, "logged_in_users": self.logged_in_users}

    def check_balance(self, user_name: str) -> float:
        """
        Check the balance of the specified user.
        """
        if user_name in self.users:
            return self.users[user_name]["balance"]
        else:
            print(f"User {user_name} does not exist!")
            return 0.0



    def add_food_delivery_order(
        self,
        username: str,
        merchant_name: str,
        items: List[Dict[str, Union[str, int]]]
    ) -> Dict[str, Union[bool, str]]:
        if username not in self.logged_in_users:
            return {
                "status": False,
                "message": f"User {username} is not logged in to the food platform"
            }

        if merchant_name not in self.merchant_list:
            return {"status": False, "message": "Merchant does not exist"}

        total_price = 0.0
        order_items = []

        for item in items:
            product_name = item.get("product")
            quantity = item.get("quantity", 1)

            if not isinstance(quantity, int) or quantity <= 0:
                return {
                    "status": False,
                    "message": f"Invalid quantity {quantity} for product {product_name}"
                }

            # Find the product price
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
                return {
                    "status": False,
                    "message": f"Product {product_name} does not exist in {merchant_name}'s menu"
                }

        # Check if the balance is sufficient
        if total_price > self.users[username]["balance"]:
            return {"status": False, "message": "Insufficient balance to place the order"}

        # Deduct the balance and create the order
        self.users[username]["balance"] -= total_price
        order = {
            "user_name": username,
            "merchant_name": merchant_name,
            "items": order_items,
            "total_price": total_price,
        }
        self.orders.append(order)
        return {
            "status": True,
            "message": f"Food delivery order successfully placed with {merchant_name}. Total amount: {total_price} yuan"
        }


    def get_products(self, merchant_name: str) -> Union[List[Dict[str, Union[str, float]]], str]:
        """
        Retrieve the product list of a specific merchant.

        :param merchant_name: Name of the merchant.
        :return: Merchant's menu list or an error message.
        """
        merchant = self.merchant_list.get(merchant_name)
        if merchant:
            return merchant["menu"]
        else:
            return {
                "status": False,
                "message": f"Merchant '{merchant_name}' does not exist"
            }


    def view_orders(self, user_name: str) -> Dict[str, Union[bool, str, List[Dict[str, Union[str, int, float]]]]]:
        """
        View all orders of a user.
        Args:
            user_name (str): Username.
        Returns:
            Dict[str, Union[bool, str, List[Dict[str, Union[str, int, float]]]]]: List of user's orders.
        """
        user_orders = [order for order in self.orders if order['user_name'] == user_name]

        if not user_orders:
            return {"status": False, "message": "User has no order records"}

        return {"status": True, "orders": user_orders}


    def search_orders(self, keyword: str) -> Dict[str, Union[bool, str, List[Dict[str, Union[str, float]]]]]:
        """
        Search orders based on a keyword.
        Args:
            keyword (str): Keyword.
        Returns:
            Dict[str, Union[bool, str, List[Dict[str, Union[str, float]]]]]: List of matching orders.
        """
        matched_orders = [
            order for order in self.orders
            if keyword.lower() in order['merchant_name'].lower()
            or any(keyword.lower() in item['product'].lower() for item in order.get('items', []))
        ]

        if not matched_orders:
            return {"status": False, "message": "No matching orders found"}

        return {"status": True, "orders": matched_orders}