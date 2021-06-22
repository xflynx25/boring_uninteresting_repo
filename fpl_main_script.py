def main():
    from Overseer import FPL_AI
    from Personalities import personalities_to_run
    for pers in personalities_to_run:
        ai = FPL_AI(**pers)
        ai.make_moves()


if __name__ == '__main__':
    main()