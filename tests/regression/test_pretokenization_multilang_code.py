#!/usr/bin/env python3
"""
Comprehensive pretokenization tests for code from various programming languages.

These tests ensure that pretokenization works correctly across different programming
languages with their unique syntax patterns, keywords, operators, and conventions.
"""

import sys
import os
import pytest

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from bpe_qwen.bpe_qwen import pretokenize_slow, pretokenize_fast


class TestPythonCode:
    """Test pretokenization of Python code snippets."""

    def test_python_function_definition(self):
        """Test Python function with type hints and docstrings."""
        code = '''def calculate_fibonacci(n: int) -> int:
    """Calculate the nth Fibonacci number using dynamic programming."""
    if n <= 1:
        return n

    dp = [0, 1]
    for i in range(2, n + 1):
        dp.append(dp[i-1] + dp[i-2])

    return dp[n]'''

        slow = pretokenize_slow(code)
        fast = pretokenize_fast(code)

        assert slow == fast, "Slow and fast tokenization should match for Python code"

        # Verify key Python elements are tokenized correctly
        tokens_str = ' '.join(slow)
        assert 'def' in tokens_str
        assert 'calculate' in tokens_str and '_fibonacci' in tokens_str
        assert 'int' in tokens_str
        assert 'return' in tokens_str

    def test_python_class_with_methods(self):
        """Test Python class definition with methods and decorators."""
        code = '''class DataProcessor:
    def __init__(self, config: dict):
        self.config = config
        self._cache = {}

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    @staticmethod
    def validate_input(data: str) -> bool:
        return isinstance(data, str) and len(data) > 0'''

        slow = pretokenize_slow(code)
        fast = pretokenize_fast(code)

        assert slow == fast, "Slow and fast tokenization should match for Python class"

    def test_python_list_comprehension(self):
        """Test Python list comprehensions and lambda functions."""
        code = '''numbers = [1, 2, 3, 4, 5]
squares = [x**2 for x in numbers if x % 2 == 0]
filtered = list(filter(lambda x: x > 10, squares))
mapped = list(map(lambda x: x * 2, filtered))'''

        slow = pretokenize_slow(code)
        fast = pretokenize_fast(code)

        assert slow == fast, "Slow and fast tokenization should match for Python comprehensions"

    def test_python_f_strings(self):
        """Test Python f-string formatting."""
        code = '''name = "Alice"
age = 30
message = f"Hello, {name}! You are {age} years old."
complex_expr = f"Result: {2 + 3 * 4:.2f}"'''

        slow = pretokenize_slow(code)
        fast = pretokenize_fast(code)

        assert slow == fast, "Slow and fast tokenization should match for Python f-strings"


class TestJavaScriptCode:
    """Test pretokenization of JavaScript code snippets."""

    def test_javascript_async_function(self):
        """Test JavaScript async/await patterns."""
        code = '''async function fetchUserData(userId) {
    try {
        const response = await fetch(`/api/users/${userId}`);
        const userData = await response.json();
        return userData;
    } catch (error) {
        console.error('Failed to fetch user data:', error);
        throw new Error('User data unavailable');
    }
}'''

        slow = pretokenize_slow(code)
        fast = pretokenize_fast(code)

        assert slow == fast, "Slow and fast tokenization should match for JavaScript async code"

    def test_javascript_arrow_functions(self):
        """Test JavaScript arrow functions and array methods."""
        code = '''const numbers = [1, 2, 3, 4, 5];
const doubled = numbers.map(x => x * 2);
const filtered = numbers.filter(x => x % 2 === 0);
const sum = numbers.reduce((acc, curr) => acc + curr, 0);'''

        slow = pretokenize_slow(code)
        fast = pretokenize_fast(code)

        assert slow == fast, "Slow and fast tokenization should match for JavaScript arrow functions"

    def test_javascript_destructuring(self):
        """Test JavaScript destructuring assignment."""
        code = '''const user = { name: 'John', age: 30, city: 'New York' };
const { name, age, ...rest } = user;
const [first, second, ...others] = [1, 2, 3, 4, 5];'''

        slow = pretokenize_slow(code)
        fast = pretokenize_fast(code)

        assert slow == fast, "Slow and fast tokenization should match for JavaScript destructuring"

    def test_javascript_template_literals(self):
        """Test JavaScript template literals with expressions."""
        code = '''const name = 'World';
const greeting = `Hello, ${name}!`;
const multiline = `
    This is a
    multiline string
    with ${name}
`;'''

        slow = pretokenize_slow(code)
        fast = pretokenize_fast(code)

        assert slow == fast, "Slow and fast tokenization should match for JavaScript template literals"


class TestJavaCode:
    """Test pretokenization of Java code snippets."""

    def test_java_class_definition(self):
        """Test Java class with generics and annotations."""
        code = '''@Entity
@Table(name = "users")
public class User<T extends Serializable> {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false)
    private String username;

    public User() {}

    public User(String username) {
        this.username = username;
    }
}'''

        slow = pretokenize_slow(code)
        fast = pretokenize_fast(code)

        assert slow == fast, "Slow and fast tokenization should match for Java class"

    def test_java_stream_operations(self):
        """Test Java 8+ stream operations."""
        code = '''List<String> names = Arrays.asList("Alice", "Bob", "Charlie");
List<String> filtered = names.stream()
    .filter(name -> name.length() > 3)
    .map(String::toUpperCase)
    .collect(Collectors.toList());'''

        slow = pretokenize_slow(code)
        fast = pretokenize_fast(code)

        assert slow == fast, "Slow and fast tokenization should match for Java streams"


class TestCppCode:
    """Test pretokenization of C++ code snippets."""

    def test_cpp_template_class(self):
        """Test C++ template class with STL containers."""
        code = '''#include <vector>
#include <algorithm>

template<typename T>
class Container {
private:
    std::vector<T> data;

public:
    void add(const T& item) {
        data.push_back(item);
    }

    void sort() {
        std::sort(data.begin(), data.end());
    }

    T& operator[](size_t index) {
        return data[index];
    }
};'''

        slow = pretokenize_slow(code)
        fast = pretokenize_fast(code)

        assert slow == fast, "Slow and fast tokenization should match for C++ template code"

    def test_cpp_lambda_expressions(self):
        """Test C++11 lambda expressions."""
        code = '''std::vector<int> numbers = {1, 2, 3, 4, 5};
auto doubled = std::transform(numbers.begin(), numbers.end(), 
    [](int x) { return x * 2; });

auto predicate = [](int x) -> bool { 
    return x % 2 == 0; 
};'''

        slow = pretokenize_slow(code)
        fast = pretokenize_fast(code)

        assert slow == fast, "Slow and fast tokenization should match for C++ lambdas"


class TestRustCode:
    """Test pretokenization of Rust code snippets."""

    def test_rust_struct_and_impl(self):
        """Test Rust struct definition with implementation."""
        code = '''#[derive(Debug, Clone)]
pub struct Person {
    pub name: String,
    pub age: u32,
}

impl Person {
    pub fn new(name: String, age: u32) -> Self {
        Person { name, age }
    }

    pub fn greet(&self) -> String {
        format!("Hello, I'm {} and I'm {} years old", self.name, self.age)
    }
}'''

        slow = pretokenize_slow(code)
        fast = pretokenize_fast(code)

        assert slow == fast, "Slow and fast tokenization should match for Rust struct"

    def test_rust_pattern_matching(self):
        """Test Rust pattern matching and Option handling."""
        code = '''fn process_option(opt: Option<i32>) -> i32 {
    match opt {
        Some(value) if value > 0 => value * 2,
        Some(value) => value,
        None => 0,
    }
}

let result = match some_function() {
    Ok(data) => data.len(),
    Err(_) => 0,
};'''

        slow = pretokenize_slow(code)
        fast = pretokenize_fast(code)

        assert slow == fast, "Slow and fast tokenization should match for Rust pattern matching"


class TestGoCode:
    """Test pretokenization of Go code snippets."""

    def test_go_struct_and_methods(self):
        """Test Go struct with methods and interfaces."""
        code = '''type User struct {
    ID       int    `json:"id"`
    Username string `json:"username"`
    Email    string `json:"email"`
}

func (u *User) Validate() error {
    if u.Username == "" {
        return errors.New("username cannot be empty")
    }
    return nil
}

func NewUser(username, email string) *User {
    return &User{
        Username: username,
        Email:    email,
    }
}'''

        slow = pretokenize_slow(code)
        fast = pretokenize_fast(code)

        assert slow == fast, "Slow and fast tokenization should match for Go struct"

    def test_go_goroutines_and_channels(self):
        """Test Go goroutines and channel operations."""
        code = '''func worker(id int, jobs <-chan int, results chan<- int) {
    for j := range jobs {
        fmt.Printf("Worker %d processing job %d\\n", id, j)
        time.Sleep(time.Second)
        results <- j * 2
    }
}

go func() {
    defer close(results)
    for i := 0; i < 5; i++ {
        jobs <- i
    }
    close(jobs)
}()'''

        slow = pretokenize_slow(code)
        fast = pretokenize_fast(code)

        assert slow == fast, "Slow and fast tokenization should match for Go concurrency"


class TestSQLCode:
    """Test pretokenization of SQL code snippets."""

    def test_complex_sql_query(self):
        """Test complex SQL query with joins and subqueries."""
        code = '''SELECT 
    u.username,
    u.email,
    COUNT(o.id) as order_count,
    SUM(o.total_amount) as total_spent
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
WHERE u.created_at >= '2023-01-01'
    AND u.status = 'active'
GROUP BY u.id, u.username, u.email
HAVING COUNT(o.id) > 0
ORDER BY total_spent DESC
LIMIT 100;'''

        slow = pretokenize_slow(code)
        fast = pretokenize_fast(code)

        assert slow == fast, "Slow and fast tokenization should match for SQL query"

    def test_sql_stored_procedure(self):
        """Test SQL stored procedure with control flow."""
        code = '''CREATE PROCEDURE UpdateUserStatus(
    IN user_id INT,
    IN new_status VARCHAR(50)
)
BEGIN
    DECLARE current_status VARCHAR(50);

    SELECT status INTO current_status 
    FROM users 
    WHERE id = user_id;

    IF current_status != new_status THEN
        UPDATE users 
        SET status = new_status, 
            updated_at = NOW()
        WHERE id = user_id;
    END IF;
END;'''

        slow = pretokenize_slow(code)
        fast = pretokenize_fast(code)

        assert slow == fast, "Slow and fast tokenization should match for SQL procedure"


class TestJSONAndYAML:
    """Test pretokenization of configuration files."""

    def test_complex_json(self):
        """Test complex JSON configuration."""
        code = '''{
  "database": {
    "host": "localhost",
    "port": 5432,
    "name": "myapp",
    "credentials": {
      "username": "admin",
      "password": "secret123"
    }
  },
  "features": {
    "authentication": true,
    "logging": {
      "level": "info",
      "file": "/var/log/app.log"
    }
  },
  "api_keys": [
    "key1",
    "key2",
    "key3"
  ]
}'''

        slow = pretokenize_slow(code)
        fast = pretokenize_fast(code)

        assert slow == fast, "Slow and fast tokenization should match for JSON"

    def test_yaml_configuration(self):
        """Test YAML configuration file."""
        code = '''version: '3.8'
services:
  web:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DEBUG=1
      - DATABASE_URL=postgresql://user:pass@db:5432/myapp
    depends_on:
      - db
      - redis

  db:
    image: postgres:13
    environment:
      POSTGRES_DB: myapp
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
    volumes:
      - postgres_data:/var/lib/postgresql/data'''

        slow = pretokenize_slow(code)
        fast = pretokenize_fast(code)

        assert slow == fast, "Slow and fast tokenization should match for YAML"


class TestRegexAndPatterns:
    """Test pretokenization of regex patterns and complex strings."""

    def test_regex_patterns(self):
        """Test various regex patterns."""
        code = '''import re

email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
phone_pattern = r'\\+?1?\\d{9,15}$'
url_pattern = r'https?://(?:[-\\w.])+(?:[:\\d]+)?(?:/(?:[\\w/_.])*(?:\\?(?:[\\w&=%.])*)?(?:#(?:\\w*))?)?'

def validate_email(email):
    return re.match(email_pattern, email) is not None'''

        slow = pretokenize_slow(code)
        fast = pretokenize_fast(code)

        assert slow == fast, "Slow and fast tokenization should match for regex patterns"

    def test_complex_string_literals(self):
        """Test complex string literals with escapes."""
        code = '''multiline_string = """
This is a multiline string
with "quotes" and 'apostrophes'
and even \\n escape sequences
"""

raw_string = r'C:\\Users\\Name\\Documents\\file.txt'
unicode_string = "Hello, 世界! 🌍"
formatted_string = f"Value: {42:>10.2f}"'''

        slow = pretokenize_slow(code)
        fast = pretokenize_fast(code)

        assert slow == fast, "Slow and fast tokenization should match for complex strings"


class TestPerformanceWithLargeCode:
    """Test pretokenization performance with larger code samples."""

    def test_large_python_module(self):
        """Test pretokenization of a large Python module."""
        # Create a substantial Python code sample
        code = '''#!/usr/bin/env python3
"""
A comprehensive data processing module with multiple classes and functions.
This module demonstrates various Python features and patterns.
"""

import os
import sys
import json
import logging
import asyncio
from typing import List, Dict, Optional, Union, Any
from dataclasses import dataclass, field
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

logger = logging.getLogger(__name__)

@dataclass
class ProcessingConfig:
    """Configuration for data processing operations."""
    batch_size: int = 1000
    max_workers: int = 4
    timeout: float = 30.0
    retry_attempts: int = 3
    output_format: str = "json"
    compression: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

class DataProcessor:
    """Main data processing class with async capabilities."""

    def __init__(self, config: ProcessingConfig):
        self.config = config
        self._cache = {}
        self._stats = {"processed": 0, "errors": 0, "skipped": 0}

    async def process_batch(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of items asynchronously."""
        results = []

        async with asyncio.TaskGroup() as tg:
            tasks = [
                tg.create_task(self._process_item(item))
                for item in items
            ]

        for task in tasks:
            try:
                result = await task
                if result:
                    results.append(result)
                    self._stats["processed"] += 1
                else:
                    self._stats["skipped"] += 1
            except Exception as e:
                logger.error(f"Error processing item: {e}")
                self._stats["errors"] += 1

        return results

    async def _process_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single item with validation and transformation."""
        if not self._validate_item(item):
            return None

        # Apply transformations
        transformed = await self._transform_item(item)

        # Add metadata
        transformed["_metadata"] = {
            "processed_at": asyncio.get_event_loop().time(),
            "processor_version": "1.0.0"
        }

        return transformed

    def _validate_item(self, item: Dict[str, Any]) -> bool:
        """Validate item structure and content."""
        required_fields = ["id", "type", "data"]
        return all(field in item for field in required_fields)

    async def _transform_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Transform item data based on type."""
        item_type = item.get("type")

        transformers = {
            "text": self._transform_text,
            "numeric": self._transform_numeric,
            "categorical": self._transform_categorical
        }

        transformer = transformers.get(item_type, self._default_transform)
        return await transformer(item)

    async def _transform_text(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Transform text data."""
        data = item["data"]
        if isinstance(data, str):
            item["data"] = {
                "original": data,
                "length": len(data),
                "words": len(data.split()),
                "normalized": data.lower().strip()
            }
        return item

    async def _transform_numeric(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Transform numeric data."""
        data = item["data"]
        if isinstance(data, (int, float)):
            item["data"] = {
                "value": data,
                "squared": data ** 2,
                "is_positive": data > 0,
                "magnitude": abs(data)
            }
        return item

    async def _transform_categorical(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Transform categorical data."""
        data = item["data"]
        if isinstance(data, str):
            categories = self._cache.get("categories", set())
            categories.add(data)
            self._cache["categories"] = categories

            item["data"] = {
                "category": data,
                "is_new": data not in categories,
                "encoded": hash(data) % 1000
            }
        return item

    async def _default_transform(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Default transformation for unknown types."""
        item["data"] = {"raw": item["data"], "type": "unknown"}
        return item

@staticmethod
def create_sample_data(count: int = 1000) -> List[Dict[str, Any]]:
    """Create sample data for testing."""
    import random

    types = ["text", "numeric", "categorical"]
    data = []

    for i in range(count):
        item_type = random.choice(types)

        if item_type == "text":
            sample_data = f"Sample text data item {i}"
        elif item_type == "numeric":
            sample_data = random.uniform(-100, 100)
        else:  # categorical
            sample_data = random.choice(["A", "B", "C", "D", "E"])

        data.append({
            "id": i,
            "type": item_type,
            "data": sample_data
        })

    return data

async def main():
    """Main execution function."""
    config = ProcessingConfig(
        batch_size=100,
        max_workers=8,
        timeout=60.0
    )

    processor = DataProcessor(config)
    sample_data = create_sample_data(1000)

    # Process in batches
    results = []
    for i in range(0, len(sample_data), config.batch_size):
        batch = sample_data[i:i + config.batch_size]
        batch_results = await processor.process_batch(batch)
        results.extend(batch_results)

        logger.info(f"Processed batch {i // config.batch_size + 1}")

    logger.info(f"Processing complete. Stats: {processor._stats}")
    return results

if __name__ == "__main__":
    asyncio.run(main())
''' * 3  # Repeat to make it larger

        slow = pretokenize_slow(code)
        fast = pretokenize_fast(code)

        assert slow == fast, "Slow and fast tokenization should match for large Python code"

        # Verify the tokenization completed successfully
        assert len(slow) > 1000, "Should produce many tokens for large code"
        assert len(fast) > 1000, "Should produce many tokens for large code"


if __name__ == "__main__":
    # Run with pytest or directly
    pytest.main([__file__, "-v"])