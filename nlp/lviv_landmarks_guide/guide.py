from nlp.lviv_landmarks_guide.query import Query


def main():
    """
    Starts the assistant bot that parses context to find the more relevant enrty to share with you
    """
    print("=== Hello, I'm your assistant on Lviv city landmarks'\n")
    while True:
        query = input("\n[Guide]: Ask me about Lviv's remarkable places (type 'exit' to end chat)\n[User]:")
        if 'exit' in query.lower():
            break
        elif query is not None:
            query = Query(query)
            answer = query.process()
            print(f'[Guide]: {answer}')
        else:
            print("**** the query shouldn't be empty ****")

    print('=== Goodbye and thank you!')


if __name__ == '__main__':
    main()
