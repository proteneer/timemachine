# (ytz): this file is left intentionally blank.
# it's a trick to allow us to do a dummy import on the client side
# that don't have custom_ops shared objects.

# on the worker side, a custom .so file for the custom_op and will
# have higher import priority than the .py file