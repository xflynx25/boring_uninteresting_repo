import private_versions.constants as constants

def add_to_orders(user):
    with open(constants.DROPBOX_PATH + 'orders.txt', 'r') as f:
        orders = f.read().splitlines()
    print('read = ', orders)
    orders.append(user)
    with open(constants.DROPBOX_PATH + 'orders.txt', 'w') as f:
        for order in orders:
            f.write(f'{order}\n')

# removes if true, returns if it was in it to start
def remove_from_orders(user):
    with open(constants.DROPBOX_PATH + 'orders.txt', 'r') as f:
        orders = f.read().splitlines()
    print('read = ', orders)
    orders.remove(user)
    with open(constants.DROPBOX_PATH + 'orders.txt', 'w') as f:
        for order in orders:
            f.write(f'{order}\n')