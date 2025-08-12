# utils.py
import random
import time
from openai import RateLimitError, APIError, Timeout

def retry_with_exponential_backoff(
    func=None,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 5,
    errors: tuple = (RateLimitError, APIError, Timeout)
):
    """Retry a function with exponential backoff."""
    def decorator(func):

        def wrapper(*args, **kwargs):

            num_retries = 0
            delay = initial_delay

            while True:
                try:
                    return func(*args, **kwargs)
                
                except errors as e:
                    num_retries += 1

                    if num_retries > max_retries:
                        raise RuntimeError(f"Máximo de {max_retries} tentativas excedido.") from e
                    
                    delay *= exponential_base * (1 + jitter * random.random())
                    
                    print(f"Erro na API da OpenAI. Tentativa {num_retries}/{max_retries}. Retentando em {delay:.2f} segundos...")
                    time.sleep(delay)
                
                except Exception as e:
                    raise e
        
        return wrapper
    
    return decorator if func is None else decorator(func)