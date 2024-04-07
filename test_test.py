from test import get_response


def test_get_response():
    # Test case 1: Test regular response
    user_input = "Hello"
    expected_output = "Hello, how can I assist you?"
    assert get_response(user_input) == expected_output

    # Test case 2: Test clarification prompt
    user_input = "Can you please provide more information?"
    expected_output = "Sure, what specific information are you looking for?"
    assert get_response(user_input) == expected_output

    # Test case 3: Test search intent
    user_input = "Search for Python tutorials"
    expected_output = "Here are the search results for 'Python tutorials':\n\n- Tutorial 1\n- Tutorial 2\n- Tutorial 3"
    assert get_response(user_input) == expected_output

    # Test case 4: Test recursion depth limit
    user_input = "Can you please provide more information?"
    expected_output = "I'm sorry, I'm having trouble understanding your query. Could you please rephrase it or provide more context?"
    assert get_response(user_input, recursion_depth=10) == expected_output

    # Test case 5: Test no specific intent
    user_input = "I need help with my computer"
    expected_output = "I'm sorry, I don't have the information you're looking for. Can you please provide more details?"
    assert get_response(user_input) == expected_output

    print("All test cases passed!")


test_get_response()
