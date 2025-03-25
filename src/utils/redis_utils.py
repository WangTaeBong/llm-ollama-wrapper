import json
import logging
from typing import List, Dict, Any, Tuple

from src.common.config_loader import ConfigLoader
from src.services.messaging.storage import RedisKeyGenerator
from src.utils.redis_manager import RedisManager

# 설정 로드
config_loader = ConfigLoader()
settings = config_loader.get_settings()

# 로거 설정
logger = logging.getLogger(__name__)


class RedisUtils:
    """
    동기 및 비동기 모드에서 Redis 작업을 수행하는 유틸리티 클래스

    이 클래스는 메시지 저장, 조회, 삭제 및 키 스캔 등 Redis 관련 작업에 필요한
    방법들을 제공합니다. 모든 메서드는 정적(static)으로 구현되어 있어 인스턴스 생성 없이
    직접 호출할 수 있습니다.

    성능 최적화를 위해 비동기 파이프라인을 사용하며, 모든 작업에 대한 예외 처리가
    구현되어 있어 Redis 연결 실패나 작업 오류 시에도 안전하게 처리됩니다.
    """

    # 디버그 모드 설정 (로깅 수준 제어용)
    _debug_mode = False

    key_generator = RedisKeyGenerator()

    @classmethod
    def set_debug_mode(cls, enabled: bool = False) -> None:
        """
        디버그 모드 활성화/비활성화 설정

        Args:
            enabled (bool): 디버그 모드 활성화 여부 (기본값: False)
        """
        cls._debug_mode = enabled

    @classmethod
    def save_message_to_redis(
            cls,
            system_info: str,
            session_id: str,
            message: Dict[str, Any]
    ) -> bool:
        """
        동기 모드로 Redis에 메시지 저장

        지정된 시스템 정보와 세션 ID를 사용하여 생성된 키에 메시지를 저장합니다.
        설정에 따라 만료 시간(TTL)을 설정할 수 있습니다.

        Args:
            system_info (str): 시스템 정보 식별자
            session_id (str): 고유 세션 식별자
            message (Dict[str, Any]): 저장할 메시지 데이터

        Returns:
            bool: 저장 성공 시 True, 실패 시 False
        """
        sync_redis = RedisManager.get_sync_connection()
        if not sync_redis:
            if cls._debug_mode:
                logger.debug("Redis connection not available for saving message (sync)")
            return False

        try:
            key = cls.key_generator.generate_key(system_info, session_id)

            # 파이프라인 사용하여 명령 일괄 처리
            with sync_redis.pipeline(transaction=False) as pipe:
                pipe.lpush(key, json.dumps(message, ensure_ascii=False))
                if settings.redis.ttl_enabled:
                    pipe.expire(key, settings.redis.ttl_time)
                pipe.execute()

            return True
        except Exception as e:
            logger.error(f"Failed to save message to Redis (sync): {e}")
            return False

    @classmethod
    async def async_save_message_to_redis(
            cls,
            system_info: str,
            session_id: str,
            message: Dict[str, Any]
    ) -> bool:
        """
        비동기 모드로 Redis에 메시지 저장

        지정된 시스템 정보와 세션 ID를 사용하여 생성된 키에 메시지를 비동기적으로 저장합니다.
        설정에 따라 만료 시간(TTL)을 설정할 수 있습니다.

        Args:
            system_info (str): 시스템 정보 식별자
            session_id (str): 고유 세션 식별자
            message (Dict[str, Any]): 저장할 메시지 데이터

        Returns:
            bool: 저장 성공 시 True, 실패 시 False
        """
        try:
            async_redis = await RedisManager.get_async_connection()
            if not async_redis:
                if cls._debug_mode:
                    logger.debug("Redis connection not available for saving message (async)")
                return False

            key = cls.key_generator.generate_key(system_info, session_id)

            # 비동기 파이프라인 사용하여 명령 일괄 처리
            async with async_redis.pipeline(transaction=False) as pipe:
                await pipe.lpush(key, json.dumps(message, ensure_ascii=False))
                if settings.redis.ttl_enabled:
                    await pipe.expire(key, settings.redis.ttl_time)
                await pipe.execute()

            return True
        except Exception as e:
            logger.error(f"Failed to save message to Redis (async): {e}")
            return False

    @classmethod
    def get_messages_from_redis(
            cls,
            system_info: str,
            session_id: str
    ) -> List[Dict[str, Any]]:
        """
        동기 모드로 Redis에서 메시지 목록 조회

        지정된 시스템 정보와 세션 ID에 해당하는 키의 메시지 목록을 조회합니다.
        설정에 정의된 개수만큼 가장 최근의 메시지를 반환합니다.

        Args:
            system_info (str): 시스템 정보 식별자
            session_id (str): 고유 세션 식별자

        Returns:
            List[Dict[str, Any]]: Redis에서 조회한 메시지 목록 (실패 시 빈 목록 반환)
        """
        sync_redis = RedisManager.get_sync_connection()
        if not sync_redis:
            if cls._debug_mode:
                logger.debug("Redis connection not available for retrieving messages (sync)")
            return []

        try:
            key = cls.key_generator.generate_key(system_info, session_id)
            count = int(settings.redis.get_message_count)

            # 메시지 목록 조회 및 JSON 디코딩
            messages = sync_redis.lrange(key, 0, count)
            return [json.loads(msg) for msg in messages]
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON format in Redis messages (sync): {e}")
            return []
        except Exception as e:
            logger.error(f"Failed to retrieve messages from Redis (sync): {e}")
            return []

    @classmethod
    async def async_get_messages_from_redis(
            cls,
            system_info: str,
            session_id: str
    ) -> List[Dict[str, Any]]:
        """
        비동기 모드로 Redis에서 메시지 목록 조회

        지정된 시스템 정보와 세션 ID에 해당하는 키의 메시지 목록을 비동기적으로 조회합니다.
        설정에 정의된 개수만큼 가장 최근의 메시지를 반환합니다.

        Args:
            system_info (str): 시스템 정보 식별자
            session_id (str): 고유 세션 식별자

        Returns:
            List[Dict[str, Any]]: Redis에서 조회한 메시지 목록 (실패 시 빈 목록 반환)
        """
        try:
            async_redis = await RedisManager.get_async_connection()
            if not async_redis:
                if cls._debug_mode:
                    logger.debug("Redis connection not available for retrieving messages (async)")
                return []

            key = cls.key_generator.generate_key(system_info, session_id)
            count = int(settings.redis.get_message_count)

            # 메시지 목록 비동기 조회 및 JSON 디코딩
            messages = await async_redis.lrange(key, 0, count)
            return [json.loads(msg) for msg in messages]
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON format in Redis messages (async): {e}")
            return []
        except Exception as e:
            logger.error(f"Failed to retrieve messages from Redis (async): {e}")
            return []

    @classmethod
    def delete_redis_key(cls, system_info: str, session_id: str) -> bool:
        """
        동기 모드로 Redis에서 특정 키 삭제

        지정된 시스템 정보와 세션 ID에 해당하는 키를 삭제합니다.

        Args:
            system_info (str): 시스템 정보 식별자
            session_id (str): 고유 세션 식별자

        Returns:
            bool: 삭제 성공 시 True, 실패 시 False
        """
        sync_redis = RedisManager.get_sync_connection()
        if not sync_redis:
            if cls._debug_mode:
                logger.debug("Redis connection not available for deleting key (sync)")
            return False

        try:
            key = cls.key_generator.generate_key(system_info, session_id)
            result = sync_redis.delete(key)
            return result > 0  # 삭제된 키의 수가 0보다 크면 성공
        except Exception as e:
            logger.error(f"Failed to delete Redis key (sync): {e}")
            return False

    @classmethod
    async def async_delete_redis_key(cls, system_info: str, session_id: str) -> bool:
        """
        비동기 모드로 Redis에서 특정 키 삭제

        지정된 시스템 정보와 세션 ID에 해당하는 키를 비동기적으로 삭제합니다.

        Args:
            system_info (str): 시스템 정보 식별자
            session_id (str): 고유 세션 식별자

        Returns:
            bool: 삭제 성공 시 True, 실패 시 False
        """
        try:
            async_redis = await RedisManager.get_async_connection()
            if not async_redis:
                if cls._debug_mode:
                    logger.debug("Redis connection not available for deleting key (async)")
                return False

            key = cls.key_generator.generate_key(system_info, session_id)
            result = await async_redis.delete(key)
            return result > 0  # 삭제된 키의 수가 0보다 크면 성공
        except Exception as e:
            logger.error(f"Failed to delete Redis key (async): {e}")
            return False

    @classmethod
    def scan_keys(cls, pattern: str, count: int = 100) -> List[str]:
        """
        동기 모드로 패턴에 일치하는 모든 Redis 키 스캔

        지정된 패턴과 일치하는 모든 Redis 키를 스캔하여 반환합니다.
        대규모 데이터베이스에서도 안전하게 키를 검색할 수 있도록 SCAN 명령을 사용합니다.

        Args:
            pattern (str): 키 매칭 패턴 (예: "prefix:*")
            count (int, optional): 각 스캔 반복에서 처리할 키 수 (기본값: 100)

        Returns:
            List[str]: 패턴과 일치하는 Redis 키 목록
        """
        sync_redis = RedisManager.get_sync_connection()
        if not sync_redis:
            if cls._debug_mode:
                logger.debug("Redis connection not available for scanning keys (sync)")
            return []

        try:
            cursor = '0'
            keys = []

            # SCAN 명령을 사용하여 큰 데이터셋에서도 안전하게 키 스캔
            while True:
                cursor, new_keys = sync_redis.scan(
                    cursor=cursor,
                    match=pattern,
                    count=count
                )
                keys.extend(new_keys)
                if cursor == '0':
                    break

            return keys
        except Exception as e:
            logger.error(f"Failed to scan keys in Redis (sync): {e}")
            return []

    @classmethod
    async def async_scan_keys(cls, pattern: str, count: int = 100) -> List[str]:
        """
        비동기 모드로 패턴에 일치하는 모든 Redis 키 스캔

        지정된 패턴과 일치하는 모든 Redis 키를 비동기적으로 스캔하여 반환합니다.
        대규모 데이터베이스에서도 안전하게 키를 검색할 수 있도록 SCAN 명령을 사용합니다.

        Args:
            pattern (str): 키 매칭 패턴 (예: "prefix:*")
            count (int, optional): 각 스캔 반복에서 처리할 키 수 (기본값: 100)

        Returns:
            List[str]: 패턴과 일치하는 Redis 키 목록
        """
        try:
            async_redis = await RedisManager.get_async_connection()
            if not async_redis:
                if cls._debug_mode:
                    logger.debug("Redis connection not available for scanning keys (async)")
                return []

            cursor = b'0'
            keys = []

            # 비동기 SCAN 명령을 사용하여 큰 데이터셋에서도 안전하게 키 스캔
            while True:
                cursor, new_keys = await async_redis.scan(
                    cursor=cursor,
                    match=pattern,
                    count=count
                )
                keys.extend(new_keys)
                if cursor == b'0':
                    break

            # 바이트 스트링을 일반 문자열로 변환 (필요한 경우)
            return [key.decode('utf-8') if isinstance(key, bytes) else key for key in keys]
        except Exception as e:
            logger.error(f"Failed to scan keys in Redis (async): {e}")
            return []

    @classmethod
    def get_key_ttl(cls, system_info: str, session_id: str) -> int:
        """
        동기 모드로 Redis 키의 남은 만료 시간(TTL) 조회

        지정된 시스템 정보와 세션 ID에 해당하는 키의 남은 만료 시간을 초 단위로 반환합니다.

        Args:
            system_info (str): 시스템 정보 식별자
            session_id (str): 고유 세션 식별자

        Returns:
            int: 키의 남은 만료 시간(초), 키가 없거나 만료 시간이 설정되지 않은 경우 -1 또는 -2
        """
        sync_redis = RedisManager.get_sync_connection()
        if not sync_redis:
            return -2  # -2는 키가 존재하지 않음을 의미

        try:
            key = cls.key_generator.generate_key(system_info, session_id)
            return sync_redis.ttl(key)
        except Exception as e:
            logger.error(f"Failed to get TTL for Redis key (sync): {e}")
            return -2

    @classmethod
    async def async_get_key_ttl(cls, system_info: str, session_id: str) -> int:
        """
        비동기 모드로 Redis 키의 남은 만료 시간(TTL) 조회

        지정된 시스템 정보와 세션 ID에 해당하는 키의 남은 만료 시간을 초 단위로 반환합니다.

        Args:
            system_info (str): 시스템 정보 식별자
            session_id (str): 고유 세션 식별자

        Returns:
            int: 키의 남은 만료 시간(초), 키가 없거나 만료 시간이 설정되지 않은 경우 -1 또는 -2
        """
        try:
            async_redis = await RedisManager.get_async_connection()
            if not async_redis:
                return -2  # -2는 키가 존재하지 않음을 의미

            key = cls.key_generator.generate_key(system_info, session_id)
            return await async_redis.ttl(key)
        except Exception as e:
            logger.error(f"Failed to get TTL for Redis key (async): {e}")
            return -2

    @classmethod
    def batch_delete_keys(cls, pattern: str) -> Tuple[int, int]:
        """
        동기 모드로 패턴에 일치하는 여러 Redis 키 일괄 삭제

        주어진 패턴과 일치하는 모든 키를 스캔하고 일괄 삭제합니다.
        대량의 키를 처리할 때 효율적입니다.

        Args:
            pattern (str): 삭제할 키 매칭 패턴 (예: "prefix:*")

        Returns:
            Tuple[int, int]: (처리한 키 개수, 성공적으로 삭제한 키 개수)
        """
        sync_redis = RedisManager.get_sync_connection()
        if not sync_redis:
            return 0, 0

        try:
            # 패턴과 일치하는 키 스캔
            keys = cls.scan_keys(pattern)
            total_keys = len(keys)

            if total_keys == 0:
                return 0, 0

            # 키가 있으면 일괄 삭제
            with sync_redis.pipeline(transaction=False) as pipe:
                for key in keys:
                    pipe.delete(key)
                results = pipe.execute()

            # 삭제 성공한 키 개수 카운트
            deleted_count = sum(1 for result in results if result > 0)
            return total_keys, deleted_count
        except Exception as e:
            logger.error(f"Failed to batch delete Redis keys (sync): {e}")
            return 0, 0

    @classmethod
    async def async_batch_delete_keys(cls, pattern: str) -> Tuple[int, int]:
        """
        비동기 모드로 패턴에 일치하는 여러 Redis 키 일괄 삭제

        주어진 패턴과 일치하는 모든 키를 비동기적으로 스캔하고 일괄 삭제합니다.
        대량의 키를 처리할 때 효율적입니다.

        Args:
            pattern (str): 삭제할 키 매칭 패턴 (예: "prefix:*")

        Returns:
            Tuple[int, int]: (처리한 키 개수, 성공적으로 삭제한 키 개수)
        """
        try:
            async_redis = await RedisManager.get_async_connection()
            if not async_redis:
                return 0, 0

            # 패턴과 일치하는 키 스캔
            keys = await cls.async_scan_keys(pattern)
            total_keys = len(keys)

            if total_keys == 0:
                return 0, 0

            # 키가 있으면 비동기 일괄 삭제
            async with async_redis.pipeline(transaction=False) as pipe:
                for key in keys:
                    await pipe.delete(key)
                results = await pipe.execute()

            # 삭제 성공한 키 개수 카운트
            deleted_count = sum(1 for result in results if result > 0)
            return total_keys, deleted_count
        except Exception as e:
            logger.error(f"Failed to batch delete Redis keys (async): {e}")
            return 0, 0
