-- triggers that decreases the quantity of an item after
-- adding a new order.
delimiter //
CREATE TRIGGER update_quantity
AFTER INSERT
ON orders
FOR EACH ROW
BEGIN
update items SET quantity = quantity - new.number WHERE items.name=new.item_name;
END //
delimiter ;
