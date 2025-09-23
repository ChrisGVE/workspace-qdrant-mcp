"""
Network-based Service Discovery

This module implements UDP multicast-based network discovery for services
on the local network. It allows services to announce themselves and discover
other services without relying on a central registry file.
"""

import asyncio
import json
import socket
import struct
import time
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Union, Callable
from loguru import logger

from .exceptions import NetworkError
from .registry import ServiceInfo

# logger imported from loguru


class DiscoveryMessageType(Enum):
    """Discovery message types"""
    DISCOVERY_REQUEST = "discovery_request"
    DISCOVERY_RESPONSE = "discovery_response"
    HEALTH_PING = "health_ping"
    SERVICE_ANNOUNCEMENT = "service_announcement"
    SERVICE_SHUTDOWN = "service_shutdown"


@dataclass
class DiscoveryMessage:
    """Network discovery message"""
    message_type: DiscoveryMessageType
    service_name: str
    timestamp: str
    payload: Dict
    auth_token: Optional[str] = None

    def to_json(self) -> str:
        """Convert to JSON string"""
        data = asdict(self)
        data['message_type'] = self.message_type.value
        return json.dumps(data)

    @classmethod
    def from_json(cls, json_str: str) -> 'DiscoveryMessage':
        """Create from JSON string"""
        data = json.loads(json_str)
        data['message_type'] = DiscoveryMessageType(data['message_type'])
        return cls(**data)


@dataclass
class DiscoveryEvent:
    """Discovery events broadcast to subscribers"""
    event_type: str
    service_name: str
    service_info: Optional[ServiceInfo] = None
    status: Optional[str] = None
    metrics: Optional[Dict[str, str]] = None


class NetworkDiscovery:
    """Network discovery manager using UDP multicast"""

    def __init__(self, 
                 multicast_address: str = "239.255.42.42",
                 port: int = 9999,
                 auth_token: Optional[str] = None):
        """Initialize network discovery"""
        
        # Validate multicast address
        try:
            addr_bytes = socket.inet_aton(multicast_address)
            # Check if it's in multicast range (224.0.0.0 to 239.255.255.255)
            first_octet = addr_bytes[0]
            if not (224 <= first_octet <= 239):
                raise NetworkError(f"{multicast_address} is not a valid multicast address")
        except socket.error:
            raise NetworkError(f"Invalid multicast address: {multicast_address}")

        self.multicast_address = multicast_address
        self.port = port
        self.auth_token = auth_token
        self.socket: Optional[socket.socket] = None
        self.running = False
        self.discovered_services: Dict[str, tuple[ServiceInfo, float]] = {}
        self.event_callbacks: List[Callable[[DiscoveryEvent], None]] = []
        self.cache_timeout = 300.0  # 5 minutes
        
        logger.info(f"Network discovery initialized for {multicast_address}:{port}")

    async def start(self) -> None:
        """Start the network discovery service"""
        logger.info(f"Starting network discovery on {self.multicast_address}:{self.port}")

        try:
            # Create UDP socket
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            # Bind to multicast address
            self.socket.bind(('', self.port))
            
            # Join multicast group
            mreq = struct.pack("4sl", socket.inet_aton(self.multicast_address), socket.INADDR_ANY)
            self.socket.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
            
            # Set socket to non-blocking
            self.socket.setblocking(False)
            
            self.running = True
            
            # Start receiver task
            asyncio.create_task(self._receive_loop())
            
            # Start cache cleanup task
            asyncio.create_task(self._cache_cleanup_loop())
            
            logger.info("Network discovery service started successfully")
            
        except Exception as e:
            raise NetworkError(f"Failed to start network discovery: {e}")

    async def stop(self) -> None:
        """Stop the network discovery service"""
        logger.info("Stopping network discovery service")
        
        self.running = False
        
        if self.socket:
            try:
                # Leave multicast group
                mreq = struct.pack("4sl", socket.inet_aton(self.multicast_address), socket.INADDR_ANY)
                self.socket.setsockopt(socket.IPPROTO_IP, socket.IP_DROP_MEMBERSHIP, mreq)
            except Exception as e:
                logger.warning(f"Error leaving multicast group: {e}")
            finally:
                self.socket.close()
                self.socket = None
        
        logger.info("Network discovery service stopped")

    async def announce_service(self, service_name: str, service_info: ServiceInfo) -> None:
        """Announce service availability"""
        logger.debug(f"Announcing service: {service_name}")
        
        message = DiscoveryMessage(
            message_type=DiscoveryMessageType.SERVICE_ANNOUNCEMENT,
            service_name=service_name,
            timestamp=datetime.now(timezone.utc).isoformat(),
            payload={"service_info": asdict(service_info)},
            auth_token=self.auth_token
        )
        
        await self._send_message(message)
        logger.debug(f"Service {service_name} announced successfully")

    async def discover_services(self, service_names: List[str], timeout: float = 10.0) -> Dict[str, ServiceInfo]:
        """Discover services on the network"""
        logger.info(f"Discovering services: {service_names}")
        
        request_id = str(uuid.uuid4())
        message = DiscoveryMessage(
            message_type=DiscoveryMessageType.DISCOVERY_REQUEST,
            service_name="discovery-client",
            timestamp=datetime.now(timezone.utc).isoformat(),
            payload={
                "service_names": service_names,
                "request_id": request_id
            },
            auth_token=self.auth_token
        )
        
        # Send discovery request
        await self._send_message(message)
        
        # Wait for responses
        start_time = time.time()
        discovered = {}
        
        while (time.time() - start_time) < timeout:
            # Check cached services
            for name, (service_info, last_seen) in list(self.discovered_services.items()):
                if (not service_names or name in service_names) and name not in discovered:
                    if time.time() - last_seen < self.cache_timeout:
                        discovered[name] = service_info
            
            # Break if we found all requested services
            if service_names and all(name in discovered for name in service_names):
                break
                
            await asyncio.sleep(0.1)
        
        logger.info(f"Discovery completed, found {len(discovered)} services")
        return discovered

    async def send_health_ping(self, service_name: str, status: str, metrics: Optional[Dict[str, str]] = None) -> None:
        """Send health ping"""
        logger.debug(f"Sending health ping for service: {service_name}")
        
        message = DiscoveryMessage(
            message_type=DiscoveryMessageType.HEALTH_PING,
            service_name=service_name,
            timestamp=datetime.now(timezone.utc).isoformat(),
            payload={
                "status": status,
                "metrics": metrics or {}
            },
            auth_token=self.auth_token
        )
        
        await self._send_message(message)

    async def announce_shutdown(self, service_name: str) -> None:
        """Announce service shutdown"""
        logger.info(f"Announcing shutdown for service: {service_name}")
        
        message = DiscoveryMessage(
            message_type=DiscoveryMessageType.SERVICE_SHUTDOWN,
            service_name=service_name,
            timestamp=datetime.now(timezone.utc).isoformat(),
            payload={},
            auth_token=self.auth_token
        )
        
        await self._send_message(message)

    def get_cached_services(self) -> Dict[str, ServiceInfo]:
        """Get currently discovered services from cache"""
        return {
            name: service_info 
            for name, (service_info, _) in self.discovered_services.items()
        }

    def subscribe_events(self, callback: Callable[[DiscoveryEvent], None]) -> None:
        """Subscribe to discovery events"""
        self.event_callbacks.append(callback)

    def unsubscribe_events(self, callback: Callable[[DiscoveryEvent], None]) -> None:
        """Unsubscribe from discovery events"""
        if callback in self.event_callbacks:
            self.event_callbacks.remove(callback)

    async def _send_message(self, message: DiscoveryMessage) -> None:
        """Send a message via multicast"""
        if not self.socket or not self.running:
            raise NetworkError("Socket not initialized")

        try:
            message_data = message.to_json().encode('utf-8')
            self.socket.sendto(message_data, (self.multicast_address, self.port))
            
            logger.debug(f"Sent {message.message_type.value} message for service {message.service_name}")
            
        except Exception as e:
            raise NetworkError(f"Failed to send message: {e}")

    async def _receive_loop(self) -> None:
        """Receive and process discovery messages"""
        while self.running:
            try:
                # Use select to check for available data
                ready = await asyncio.get_event_loop().run_in_executor(
                    None, self._wait_for_data, 0.1
                )
                
                if ready and self.socket:
                    data, addr = self.socket.recvfrom(4096)
                    try:
                        message = DiscoveryMessage.from_json(data.decode('utf-8'))
                        await self._process_message(message, addr)
                    except (json.JSONDecodeError, ValueError) as e:
                        logger.warning(f"Invalid message from {addr}: {e}")
                        
            except Exception as e:
                if self.running:  # Only log if we're supposed to be running
                    logger.warning(f"Error in receive loop: {e}")
                    
            await asyncio.sleep(0.001)  # Small delay to prevent busy waiting

    def _wait_for_data(self, timeout: float) -> bool:
        """Wait for socket data with timeout"""
        if not self.socket:
            return False
            
        import select
        ready, _, _ = select.select([self.socket], [], [], timeout)
        return bool(ready)

    async def _process_message(self, message: DiscoveryMessage, sender_addr: tuple) -> None:
        """Process received discovery message"""
        logger.debug(f"Processing {message.message_type.value} message from {sender_addr} for service {message.service_name}")

        # Verify authentication token if required
        if self.auth_token and message.auth_token != self.auth_token:
            logger.warning(f"Authentication failed for message from {sender_addr}")
            return

        try:
            if message.message_type == DiscoveryMessageType.SERVICE_ANNOUNCEMENT:
                service_info_data = message.payload.get("service_info", {})
                if service_info_data:
                    # Convert status back to enum if it's a string
                    if "status" in service_info_data and isinstance(service_info_data["status"], str):
                        from .registry import ServiceStatus
                        service_info_data["status"] = ServiceStatus(service_info_data["status"])
                    
                    service_info = ServiceInfo(**service_info_data)
                    self._cache_discovered_service(message.service_name, service_info)
                    
                    event = DiscoveryEvent(
                        event_type="service_discovered",
                        service_name=message.service_name,
                        service_info=service_info
                    )
                    self._broadcast_event(event)

            elif message.message_type == DiscoveryMessageType.DISCOVERY_REQUEST:
                # TODO: Respond with our own service information if applicable
                logger.debug(f"Received discovery request from {sender_addr}")

            elif message.message_type == DiscoveryMessageType.HEALTH_PING:
                status = message.payload.get("status", "unknown")
                metrics = message.payload.get("metrics", {})
                
                event = DiscoveryEvent(
                    event_type="health_ping",
                    service_name=message.service_name,
                    status=status,
                    metrics=metrics
                )
                self._broadcast_event(event)

            elif message.message_type == DiscoveryMessageType.SERVICE_SHUTDOWN:
                self._remove_cached_service(message.service_name)
                
                event = DiscoveryEvent(
                    event_type="service_lost",
                    service_name=message.service_name
                )
                self._broadcast_event(event)

        except Exception as e:
            logger.error(f"Error processing message: {e}")

    def _cache_discovered_service(self, service_name: str, service_info: ServiceInfo) -> None:
        """Cache a discovered service"""
        self.discovered_services[service_name] = (service_info, time.time())

    def _remove_cached_service(self, service_name: str) -> None:
        """Remove cached service"""
        self.discovered_services.pop(service_name, None)

    def _broadcast_event(self, event: DiscoveryEvent) -> None:
        """Broadcast event to all subscribers"""
        for callback in self.event_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Error in event callback: {e}")

    async def _cache_cleanup_loop(self) -> None:
        """Cleanup expired cache entries"""
        while self.running:
            try:
                current_time = time.time()
                expired_services = []
                
                for service_name, (_, last_seen) in list(self.discovered_services.items()):
                    if current_time - last_seen > self.cache_timeout:
                        expired_services.append(service_name)
                
                for service_name in expired_services:
                    self._remove_cached_service(service_name)
                    event = DiscoveryEvent(
                        event_type="service_lost",
                        service_name=service_name
                    )
                    self._broadcast_event(event)
                
                if expired_services:
                    logger.debug(f"Cleaned up {len(expired_services)} expired service entries")
                    
            except Exception as e:
                logger.error(f"Error in cache cleanup: {e}")
                
            await asyncio.sleep(60)  # Clean every minute