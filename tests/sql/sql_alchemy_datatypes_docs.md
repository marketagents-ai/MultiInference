The Type Hierarchy
SQLAlchemy provides abstractions for most common database data types, as well as several techniques for customization of datatypes.

Database types are represented using Python classes, all of which ultimately extend from the base type class known as TypeEngine. There are two general categories of datatypes, each of which express themselves within the typing hierarchy in different ways. The category used by an individual datatype class can be identified based on the use of two different naming conventions, which are “CamelCase” and “UPPERCASE”.

See also

Setting up MetaData with Table objects - in the SQLAlchemy Unified Tutorial. Illustrates the most rudimental use of TypeEngine type objects to define Table metadata and introduces the concept of type objects in tutorial form.

The “CamelCase” datatypes
The rudimental types have “CamelCase” names such as String, Numeric, Integer, and DateTime. All of the immediate subclasses of TypeEngine are “CamelCase” types. The “CamelCase” types are to the greatest degree possible database agnostic, meaning they can all be used on any database backend where they will behave in such a way as appropriate to that backend in order to produce the desired behavior.

An example of a straightforward “CamelCase” datatype is String. On most backends, using this datatype in a table specification will correspond to the VARCHAR database type being used on the target backend, delivering string values to and from the database, as in the example below:

from sqlalchemy import MetaData
from sqlalchemy import Table, Column, Integer, String

metadata_obj = MetaData()

user = Table(
    "user",
    metadata_obj,
    Column("user_name", String, primary_key=True),
    Column("email_address", String(60)),
)
When using a particular TypeEngine class in a Table definition or in any SQL expression overall, if no arguments are required it may be passed as the class itself, that is, without instantiating it with (). If arguments are needed, such as the length argument of 60 in the "email_address" column above, the type may be instantiated.

Another “CamelCase” datatype that expresses more backend-specific behavior is the Boolean datatype. Unlike String, which represents a string datatype that all databases have, not every backend has a real “boolean” datatype; some make use of integers or BIT values 0 and 1, some have boolean literal constants true and false while others dont. For this datatype, Boolean may render BOOLEAN on a backend such as PostgreSQL, BIT on the MySQL backend and SMALLINT on Oracle Database. As data is sent and received from the database using this type, based on the dialect in use it may be interpreting Python numeric or boolean values.

The typical SQLAlchemy application will likely wish to use primarily “CamelCase” types in the general case, as they will generally provide the best basic behavior and be automatically portable to all backends.

Reference for the general set of “CamelCase” datatypes is below at Generic “CamelCase” Types.

The “UPPERCASE” datatypes
In contrast to the “CamelCase” types are the “UPPERCASE” datatypes. These datatypes are always inherited from a particular “CamelCase” datatype, and always represent an exact datatype. When using an “UPPERCASE” datatype, the name of the type is always rendered exactly as given, without regard for whether or not the current backend supports it. Therefore the use of “UPPERCASE” types in a SQLAlchemy application indicates that specific datatypes are required, which then implies that the application would normally, without additional steps taken, be limited to those backends which use the type exactly as given. Examples of UPPERCASE types include VARCHAR, NUMERIC, INTEGER, and TIMESTAMP, which inherit directly from the previously mentioned “CamelCase” types String, Numeric, Integer, and DateTime, respectively.

The “UPPERCASE” datatypes that are part of sqlalchemy.types are common SQL types that typically expect to be available on at least two backends if not more.

Reference for the general set of “UPPERCASE” datatypes is below at SQL Standard and Multiple Vendor “UPPERCASE” Types.

Backend-specific “UPPERCASE” datatypes
Most databases also have their own datatypes that are either fully specific to those databases, or add additional arguments that are specific to those databases. For these datatypes, specific SQLAlchemy dialects provide backend-specific “UPPERCASE” datatypes, for a SQL type that has no analogue on other backends. Examples of backend-specific uppercase datatypes include PostgreSQL’s JSONB, SQL Server’s IMAGE and MySQL’s TINYTEXT.

Specific backends may also include “UPPERCASE” datatypes that extend the arguments available from that same “UPPERCASE” datatype as found in the sqlalchemy.types module. An example is when creating a MySQL string datatype, one might want to specify MySQL-specific arguments such as charset or national, which are available from the MySQL version of VARCHAR as the MySQL-only parameters VARCHAR.charset and VARCHAR.national.

API documentation for backend-specific types are in the dialect-specific documentation, listed at Dialects.

Using “UPPERCASE” and Backend-specific types for multiple backends
Reviewing the presence of “UPPERCASE” and “CamelCase” types leads to the natural use case of how to make use of “UPPERCASE” datatypes for backend-specific options, but only when that backend is in use. To tie together the database-agnostic “CamelCase” and backend-specific “UPPERCASE” systems, one makes use of the TypeEngine.with_variant() method in order to compose types together to work with specific behaviors on specific backends.

Such as, to use the String datatype, but when running on MySQL to make use of the VARCHAR.charset parameter of VARCHAR when the table is created on MySQL or MariaDB, TypeEngine.with_variant() may be used as below:

from sqlalchemy import MetaData
from sqlalchemy import Table, Column, Integer, String
from sqlalchemy.dialects.mysql import VARCHAR

metadata_obj = MetaData()

user = Table(
    "user",
    metadata_obj,
    Column("user_name", String(100), primary_key=True),
    Column(
        "bio",
        String(255).with_variant(VARCHAR(255, charset="utf8"), "mysql", "mariadb"),
    ),
)
In the above table definition, the "bio" column will have string-behaviors on all backends. On most backends it will render in DDL as VARCHAR. However on MySQL and MariaDB (indicated by database URLs that start with mysql or mariadb), it will render as VARCHAR(255) CHARACTER SET utf8.

See also

TypeEngine.with_variant() - additional usage examples and notes

Generic “CamelCase” Types
Generic types specify a column that can read, write and store a particular type of Python data. SQLAlchemy will choose the best database column type available on the target database when issuing a CREATE TABLE statement. For complete control over which column type is emitted in CREATE TABLE, such as VARCHAR see SQL Standard and Multiple Vendor “UPPERCASE” Types and the other sections of this chapter.

Object Name	Description
BigInteger

A type for bigger int integers.

Boolean

A bool datatype.

Date

A type for datetime.date() objects.

DateTime

A type for datetime.datetime() objects.

Double

A type for double FLOAT floating point types.

Enum

Generic Enum Type.

Float

Type representing floating point types, such as FLOAT or REAL.

Integer

A type for int integers.

Interval

A type for datetime.timedelta() objects.

LargeBinary

A type for large binary byte data.

MatchType

Refers to the return type of the MATCH operator.

Numeric

Base for non-integer numeric types, such as NUMERIC, FLOAT, DECIMAL, and other variants.

PickleType

Holds Python objects, which are serialized using pickle.

SchemaType

Add capabilities to a type which allow for schema-level DDL to be associated with a type.

SmallInteger

A type for smaller int integers.

String

The base for all string and character types.

Text

A variably sized string type.

Time

A type for datetime.time() objects.

Unicode

A variable length Unicode string type.

UnicodeText

An unbounded-length Unicode string type.

Uuid

Represent a database agnostic UUID datatype.

class sqlalchemy.types.BigInteger
A type for bigger int integers.

Typically generates a BIGINT in DDL, and otherwise acts like a normal Integer on the Python side.

Class signature

class sqlalchemy.types.BigInteger (sqlalchemy.types.Integer)

class sqlalchemy.types.Boolean
A bool datatype.

Boolean typically uses BOOLEAN or SMALLINT on the DDL side, and on the Python side deals in True or False.

The Boolean datatype currently has two levels of assertion that the values persisted are simple true/false values. For all backends, only the Python values None, True, False, 1 or 0 are accepted as parameter values. For those backends that don’t support a “native boolean” datatype, an option exists to also create a CHECK constraint on the target column

Changed in version 1.2: the Boolean datatype now asserts that incoming Python values are already in pure boolean form.

Members

__init__(), bind_processor(), literal_processor(), python_type, result_processor()

Class signature

class sqlalchemy.types.Boolean (sqlalchemy.types.SchemaType, sqlalchemy.types.Emulated, sqlalchemy.types.TypeEngine)

method sqlalchemy.types.Boolean.__init__(create_constraint: bool = False, name: str | None = None, _create_events: bool = True, _adapted_from: SchemaType | None = None)
Construct a Boolean.

Parameters:
create_constraint –

defaults to False. If the boolean is generated as an int/smallint, also create a CHECK constraint on the table that ensures 1 or 0 as a value.

Note

it is strongly recommended that the CHECK constraint have an explicit name in order to support schema-management concerns. This can be established either by setting the Boolean.name parameter or by setting up an appropriate naming convention; see Configuring Constraint Naming Conventions for background.

Changed in version 1.4: - this flag now defaults to False, meaning no CHECK constraint is generated for a non-native enumerated type.

name – if a CHECK constraint is generated, specify the name of the constraint.

method sqlalchemy.types.Boolean.bind_processor(dialect)
Return a conversion function for processing bind values.

Returns a callable which will receive a bind parameter value as the sole positional argument and will return a value to send to the DB-API.

If processing is not necessary, the method should return None.

Note

This method is only called relative to a dialect specific type object, which is often private to a dialect in use and is not the same type object as the public facing one, which means it’s not feasible to subclass a TypeEngine class in order to provide an alternate TypeEngine.bind_processor() method, unless subclassing the UserDefinedType class explicitly.

To provide alternate behavior for TypeEngine.bind_processor(), implement a TypeDecorator class and provide an implementation of TypeDecorator.process_bind_param().

See also

Augmenting Existing Types

Parameters:
dialect – Dialect instance in use.

method sqlalchemy.types.Boolean.literal_processor(dialect)
Return a conversion function for processing literal values that are to be rendered directly without using binds.

This function is used when the compiler makes use of the “literal_binds” flag, typically used in DDL generation as well as in certain scenarios where backends don’t accept bound parameters.

Returns a callable which will receive a literal Python value as the sole positional argument and will return a string representation to be rendered in a SQL statement.

Note

This method is only called relative to a dialect specific type object, which is often private to a dialect in use and is not the same type object as the public facing one, which means it’s not feasible to subclass a TypeEngine class in order to provide an alternate TypeEngine.literal_processor() method, unless subclassing the UserDefinedType class explicitly.

To provide alternate behavior for TypeEngine.literal_processor(), implement a TypeDecorator class and provide an implementation of TypeDecorator.process_literal_param().

See also

Augmenting Existing Types

attribute sqlalchemy.types.Boolean.python_type
method sqlalchemy.types.Boolean.result_processor(dialect, coltype)
Return a conversion function for processing result row values.

Returns a callable which will receive a result row column value as the sole positional argument and will return a value to return to the user.

If processing is not necessary, the method should return None.

Note

This method is only called relative to a dialect specific type object, which is often private to a dialect in use and is not the same type object as the public facing one, which means it’s not feasible to subclass a TypeEngine class in order to provide an alternate TypeEngine.result_processor() method, unless subclassing the UserDefinedType class explicitly.

To provide alternate behavior for TypeEngine.result_processor(), implement a TypeDecorator class and provide an implementation of TypeDecorator.process_result_value().

See also

Augmenting Existing Types

Parameters:
dialect – Dialect instance in use.

coltype – DBAPI coltype argument received in cursor.description.

class sqlalchemy.types.Date
A type for datetime.date() objects.

Members

get_dbapi_type(), literal_processor(), python_type

Class signature

class sqlalchemy.types.Date (sqlalchemy.types._RenderISO8601NoT, sqlalchemy.types.HasExpressionLookup, sqlalchemy.types.TypeEngine)

method sqlalchemy.types.Date.get_dbapi_type(dbapi)
Return the corresponding type object from the underlying DB-API, if any.

This can be useful for calling setinputsizes(), for example.

method sqlalchemy.types.Date.literal_processor(dialect)
Return a conversion function for processing literal values that are to be rendered directly without using binds.

This function is used when the compiler makes use of the “literal_binds” flag, typically used in DDL generation as well as in certain scenarios where backends don’t accept bound parameters.

Returns a callable which will receive a literal Python value as the sole positional argument and will return a string representation to be rendered in a SQL statement.

Note

This method is only called relative to a dialect specific type object, which is often private to a dialect in use and is not the same type object as the public facing one, which means it’s not feasible to subclass a TypeEngine class in order to provide an alternate TypeEngine.literal_processor() method, unless subclassing the UserDefinedType class explicitly.

To provide alternate behavior for TypeEngine.literal_processor(), implement a TypeDecorator class and provide an implementation of TypeDecorator.process_literal_param().

See also

Augmenting Existing Types

attribute sqlalchemy.types.Date.python_type
class sqlalchemy.types.DateTime
A type for datetime.datetime() objects.

Date and time types return objects from the Python datetime module. Most DBAPIs have built in support for the datetime module, with the noted exception of SQLite. In the case of SQLite, date and time types are stored as strings which are then converted back to datetime objects when rows are returned.

For the time representation within the datetime type, some backends include additional options, such as timezone support and fractional seconds support. For fractional seconds, use the dialect-specific datatype, such as TIME. For timezone support, use at least the TIMESTAMP datatype, if not the dialect-specific datatype object.

Members

__init__(), get_dbapi_type(), literal_processor(), python_type

Class signature

class sqlalchemy.types.DateTime (sqlalchemy.types._RenderISO8601NoT, sqlalchemy.types.HasExpressionLookup, sqlalchemy.types.TypeEngine)

method sqlalchemy.types.DateTime.__init__(timezone: bool = False)
Construct a new DateTime.

Parameters:
timezone – boolean. Indicates that the datetime type should enable timezone support, if available on the base date/time-holding type only. It is recommended to make use of the TIMESTAMP datatype directly when using this flag, as some databases include separate generic date/time-holding types distinct from the timezone-capable TIMESTAMP datatype, such as Oracle Database.

method sqlalchemy.types.DateTime.get_dbapi_type(dbapi)
Return the corresponding type object from the underlying DB-API, if any.

This can be useful for calling setinputsizes(), for example.

method sqlalchemy.types.DateTime.literal_processor(dialect)
Return a conversion function for processing literal values that are to be rendered directly without using binds.

This function is used when the compiler makes use of the “literal_binds” flag, typically used in DDL generation as well as in certain scenarios where backends don’t accept bound parameters.

Returns a callable which will receive a literal Python value as the sole positional argument and will return a string representation to be rendered in a SQL statement.

Note

This method is only called relative to a dialect specific type object, which is often private to a dialect in use and is not the same type object as the public facing one, which means it’s not feasible to subclass a TypeEngine class in order to provide an alternate TypeEngine.literal_processor() method, unless subclassing the UserDefinedType class explicitly.

To provide alternate behavior for TypeEngine.literal_processor(), implement a TypeDecorator class and provide an implementation of TypeDecorator.process_literal_param().

See also

Augmenting Existing Types

attribute sqlalchemy.types.DateTime.python_type
class sqlalchemy.types.Enum
Generic Enum Type.

The Enum type provides a set of possible string values which the column is constrained towards.

The Enum type will make use of the backend’s native “ENUM” type if one is available; otherwise, it uses a VARCHAR datatype. An option also exists to automatically produce a CHECK constraint when the VARCHAR (so called “non-native”) variant is produced; see the Enum.create_constraint flag.

The Enum type also provides in-Python validation of string values during both read and write operations. When reading a value from the database in a result set, the string value is always checked against the list of possible values and a LookupError is raised if no match is found. When passing a value to the database as a plain string within a SQL statement, if the Enum.validate_strings parameter is set to True, a LookupError is raised for any string value that’s not located in the given list of possible values; note that this impacts usage of LIKE expressions with enumerated values (an unusual use case).

The source of enumerated values may be a list of string values, or alternatively a PEP-435-compliant enumerated class. For the purposes of the Enum datatype, this class need only provide a __members__ method.

When using an enumerated class, the enumerated objects are used both for input and output, rather than strings as is the case with a plain-string enumerated type:

import enum
from sqlalchemy import Enum


class MyEnum(enum.Enum):
    one = 1
    two = 2
    three = 3


t = Table("data", MetaData(), Column("value", Enum(MyEnum)))

connection.execute(t.insert(), {"value": MyEnum.two})
assert connection.scalar(t.select()) is MyEnum.two
Above, the string names of each element, e.g. “one”, “two”, “three”, are persisted to the database; the values of the Python Enum, here indicated as integers, are not used; the value of each enum can therefore be any kind of Python object whether or not it is persistable.

In order to persist the values and not the names, the Enum.values_callable parameter may be used. The value of this parameter is a user-supplied callable, which is intended to be used with a PEP-435-compliant enumerated class and returns a list of string values to be persisted. For a simple enumeration that uses string values, a callable such as lambda x: [e.value for e in x] is sufficient.

See also

Using Python Enum or pep-586 Literal types in the type map - background on using the Enum datatype with the ORM’s ORM Annotated Declarative feature.

ENUM - PostgreSQL-specific type, which has additional functionality.

ENUM - MySQL-specific type

Members

__init__(), create(), drop()

Class signature

class sqlalchemy.types.Enum (sqlalchemy.types.String, sqlalchemy.types.SchemaType, sqlalchemy.types.Emulated, sqlalchemy.types.TypeEngine)

method sqlalchemy.types.Enum.__init__(*enums: object, **kw: Any)
Construct an enum.

Keyword arguments which don’t apply to a specific backend are ignored by that backend.

Parameters:
*enums – either exactly one PEP-435 compliant enumerated type or one or more string labels.

create_constraint –

defaults to False. When creating a non-native enumerated type, also build a CHECK constraint on the database against the valid values.

Note

it is strongly recommended that the CHECK constraint have an explicit name in order to support schema-management concerns. This can be established either by setting the Enum.name parameter or by setting up an appropriate naming convention; see Configuring Constraint Naming Conventions for background.

Changed in version 1.4: - this flag now defaults to False, meaning no CHECK constraint is generated for a non-native enumerated type.

metadata –

Associate this type directly with a MetaData object. For types that exist on the target database as an independent schema construct (PostgreSQL), this type will be created and dropped within create_all() and drop_all() operations. If the type is not associated with any MetaData object, it will associate itself with each Table in which it is used, and will be created when any of those individual tables are created, after a check is performed for its existence. The type is only dropped when drop_all() is called for that Table object’s metadata, however.

The value of the MetaData.schema parameter of the MetaData object, if set, will be used as the default value of the Enum.schema on this object if an explicit value is not otherwise supplied.

Changed in version 1.4.12: Enum inherits the MetaData.schema parameter of the MetaData object if present, when passed using the Enum.metadata parameter.

name – The name of this type. This is required for PostgreSQL and any future supported database which requires an explicitly named type, or an explicitly named constraint in order to generate the type and/or a table that uses it. If a PEP-435 enumerated class was used, its name (converted to lower case) is used by default.

native_enum – Use the database’s native ENUM type when available. Defaults to True. When False, uses VARCHAR + check constraint for all backends. When False, the VARCHAR length can be controlled with Enum.length; currently “length” is ignored if native_enum=True.

length –

Allows specifying a custom length for the VARCHAR when a non-native enumeration datatype is used. By default it uses the length of the longest value.

Changed in version 2.0.0: The Enum.length parameter is used unconditionally for VARCHAR rendering regardless of the Enum.native_enum parameter, for those backends where VARCHAR is used for enumerated datatypes.

schema –

Schema name of this type. For types that exist on the target database as an independent schema construct (PostgreSQL), this parameter specifies the named schema in which the type is present.

If not present, the schema name will be taken from the MetaData collection if passed as Enum.metadata, for a MetaData that includes the MetaData.schema parameter.

Changed in version 1.4.12: Enum inherits the MetaData.schema parameter of the MetaData object if present, when passed using the Enum.metadata parameter.

Otherwise, if the Enum.inherit_schema flag is set to True, the schema will be inherited from the associated Table object if any; when Enum.inherit_schema is at its default of False, the owning table’s schema is not used.

quote – Set explicit quoting preferences for the type’s name.

inherit_schema – When True, the “schema” from the owning Table will be copied to the “schema” attribute of this Enum, replacing whatever value was passed for the schema attribute. This also takes effect when using the Table.to_metadata() operation.

validate_strings – when True, string values that are being passed to the database in a SQL statement will be checked for validity against the list of enumerated values. Unrecognized values will result in a LookupError being raised.

values_callable –

A callable which will be passed the PEP-435 compliant enumerated type, which should then return a list of string values to be persisted. This allows for alternate usages such as using the string value of an enum to be persisted to the database instead of its name. The callable must return the values to be persisted in the same order as iterating through the Enum’s __member__ attribute. For example lambda x: [i.value for i in x].

New in version 1.2.3.

sort_key_function –

a Python callable which may be used as the “key” argument in the Python sorted() built-in. The SQLAlchemy ORM requires that primary key columns which are mapped must be sortable in some way. When using an unsortable enumeration object such as a Python 3 Enum object, this parameter may be used to set a default sort key function for the objects. By default, the database value of the enumeration is used as the sorting function.

New in version 1.3.8.

omit_aliases –

A boolean that when true will remove aliases from pep 435 enums. defaults to True.

Changed in version 2.0: This parameter now defaults to True.

method sqlalchemy.types.Enum.create(bind, checkfirst=False)
inherited from the SchemaType.create() method of SchemaType

Issue CREATE DDL for this type, if applicable.

method sqlalchemy.types.Enum.drop(bind, checkfirst=False)
inherited from the SchemaType.drop() method of SchemaType

Issue DROP DDL for this type, if applicable.

class sqlalchemy.types.Double
A type for double FLOAT floating point types.

Typically generates a DOUBLE or DOUBLE_PRECISION in DDL, and otherwise acts like a normal Float on the Python side.

New in version 2.0.

Class signature

class sqlalchemy.types.Double (sqlalchemy.types.Float)

class sqlalchemy.types.Float
Type representing floating point types, such as FLOAT or REAL.

This type returns Python float objects by default, unless the Float.asdecimal flag is set to True, in which case they are coerced to decimal.Decimal objects.

When a Float.precision is not provided in a Float type some backend may compile this type as an 8 bytes / 64 bit float datatype. To use a 4 bytes / 32 bit float datatype a precision <= 24 can usually be provided or the REAL type can be used. This is known to be the case in the PostgreSQL and MSSQL dialects that render the type as FLOAT that’s in both an alias of DOUBLE PRECISION. Other third party dialects may have similar behavior.

Members

__init__(), result_processor()

Class signature

class sqlalchemy.types.Float (sqlalchemy.types.Numeric)

method sqlalchemy.types.Float.__init__(precision: int | None = None, asdecimal: bool = False, decimal_return_scale: int | None = None)
Construct a Float.

Parameters:
precision –

the numeric precision for use in DDL CREATE TABLE. Backends should attempt to ensure this precision indicates a number of digits for the generic Float datatype.

Note

For the Oracle Database backend, the Float.precision parameter is not accepted when rendering DDL, as Oracle Database does not support float precision specified as a number of decimal places. Instead, use the Oracle Database-specific FLOAT datatype and specify the FLOAT.binary_precision parameter. This is new in version 2.0 of SQLAlchemy.

To create a database agnostic Float that separately specifies binary precision for Oracle Database, use TypeEngine.with_variant() as follows:

from sqlalchemy import Column
from sqlalchemy import Float
from sqlalchemy.dialects import oracle

Column(
    "float_data",
    Float(5).with_variant(oracle.FLOAT(binary_precision=16), "oracle"),
)
asdecimal – the same flag as that of Numeric, but defaults to False. Note that setting this flag to True results in floating point conversion.

decimal_return_scale – Default scale to use when converting from floats to Python decimals. Floating point values will typically be much longer due to decimal inaccuracy, and most floating point database types don’t have a notion of “scale”, so by default the float type looks for the first ten decimal places when converting. Specifying this value will override that length. Note that the MySQL float types, which do include “scale”, will use “scale” as the default for decimal_return_scale, if not otherwise specified.

method sqlalchemy.types.Float.result_processor(dialect, coltype)
Return a conversion function for processing result row values.

Returns a callable which will receive a result row column value as the sole positional argument and will return a value to return to the user.

If processing is not necessary, the method should return None.

Note

This method is only called relative to a dialect specific type object, which is often private to a dialect in use and is not the same type object as the public facing one, which means it’s not feasible to subclass a TypeEngine class in order to provide an alternate TypeEngine.result_processor() method, unless subclassing the UserDefinedType class explicitly.

To provide alternate behavior for TypeEngine.result_processor(), implement a TypeDecorator class and provide an implementation of TypeDecorator.process_result_value().

See also

Augmenting Existing Types

Parameters:
dialect – Dialect instance in use.

coltype – DBAPI coltype argument received in cursor.description.

class sqlalchemy.types.Integer
A type for int integers.

Members

get_dbapi_type(), literal_processor(), python_type

Class signature

class sqlalchemy.types.Integer (sqlalchemy.types.HasExpressionLookup, sqlalchemy.types.TypeEngine)

method sqlalchemy.types.Integer.get_dbapi_type(dbapi)
Return the corresponding type object from the underlying DB-API, if any.

This can be useful for calling setinputsizes(), for example.

method sqlalchemy.types.Integer.literal_processor(dialect)
Return a conversion function for processing literal values that are to be rendered directly without using binds.

This function is used when the compiler makes use of the “literal_binds” flag, typically used in DDL generation as well as in certain scenarios where backends don’t accept bound parameters.

Returns a callable which will receive a literal Python value as the sole positional argument and will return a string representation to be rendered in a SQL statement.

Note

This method is only called relative to a dialect specific type object, which is often private to a dialect in use and is not the same type object as the public facing one, which means it’s not feasible to subclass a TypeEngine class in order to provide an alternate TypeEngine.literal_processor() method, unless subclassing the UserDefinedType class explicitly.

To provide alternate behavior for TypeEngine.literal_processor(), implement a TypeDecorator class and provide an implementation of TypeDecorator.process_literal_param().

See also

Augmenting Existing Types

attribute sqlalchemy.types.Integer.python_type
class sqlalchemy.types.Interval
A type for datetime.timedelta() objects.

The Interval type deals with datetime.timedelta objects. In PostgreSQL and Oracle Database, the native INTERVAL type is used; for others, the value is stored as a date which is relative to the “epoch” (Jan. 1, 1970).

Note that the Interval type does not currently provide date arithmetic operations on platforms which do not support interval types natively. Such operations usually require transformation of both sides of the expression (such as, conversion of both sides into integer epoch values first) which currently is a manual procedure (such as via expression.func).

Members

__init__(), adapt_to_emulated(), bind_processor(), cache_ok, coerce_compared_value(), comparator_factory, impl, python_type, result_processor()

Class signature

class sqlalchemy.types.Interval (sqlalchemy.types.Emulated, sqlalchemy.types._AbstractInterval, sqlalchemy.types.TypeDecorator)

class Comparator
Class signature

class sqlalchemy.types.Interval.Comparator (sqlalchemy.types.Comparator, sqlalchemy.types.Comparator)

method sqlalchemy.types.Interval.__init__(native: bool = True, second_precision: int | None = None, day_precision: int | None = None)
Construct an Interval object.

Parameters:
native – when True, use the actual INTERVAL type provided by the database, if supported (currently PostgreSQL, Oracle Database). Otherwise, represent the interval data as an epoch value regardless.

second_precision – For native interval types which support a “fractional seconds precision” parameter, i.e. Oracle Database and PostgreSQL

day_precision – for native interval types which support a “day precision” parameter, i.e. Oracle Database.

method sqlalchemy.types.Interval.adapt_to_emulated(impltype, **kw)
Given an impl class, adapt this type to the impl assuming “emulated”.

The impl should also be an “emulated” version of this type, most likely the same class as this type itself.

e.g.: sqltypes.Enum adapts to the Enum class.

method sqlalchemy.types.Interval.bind_processor(dialect: Dialect) → _BindProcessorType[dt.timedelta]
Return a conversion function for processing bind values.

Returns a callable which will receive a bind parameter value as the sole positional argument and will return a value to send to the DB-API.

If processing is not necessary, the method should return None.

Note

This method is only called relative to a dialect specific type object, which is often private to a dialect in use and is not the same type object as the public facing one, which means it’s not feasible to subclass a TypeEngine class in order to provide an alternate TypeEngine.bind_processor() method, unless subclassing the UserDefinedType class explicitly.

To provide alternate behavior for TypeEngine.bind_processor(), implement a TypeDecorator class and provide an implementation of TypeDecorator.process_bind_param().

See also

Augmenting Existing Types

Parameters:
dialect – Dialect instance in use.

attribute sqlalchemy.types.Interval.cache_ok: bool | None = True
Indicate if statements using this ExternalType are “safe to cache”.

The default value None will emit a warning and then not allow caching of a statement which includes this type. Set to False to disable statements using this type from being cached at all without a warning. When set to True, the object’s class and selected elements from its state will be used as part of the cache key. For example, using a TypeDecorator:

class MyType(TypeDecorator):
    impl = String

    cache_ok = True

    def __init__(self, choices):
        self.choices = tuple(choices)
        self.internal_only = True
The cache key for the above type would be equivalent to:

>>> MyType(["a", "b", "c"])._static_cache_key
(<class '__main__.MyType'>, ('choices', ('a', 'b', 'c')))
The caching scheme will extract attributes from the type that correspond to the names of parameters in the __init__() method. Above, the “choices” attribute becomes part of the cache key but “internal_only” does not, because there is no parameter named “internal_only”.

The requirements for cacheable elements is that they are hashable and also that they indicate the same SQL rendered for expressions using this type every time for a given cache value.

To accommodate for datatypes that refer to unhashable structures such as dictionaries, sets and lists, these objects can be made “cacheable” by assigning hashable structures to the attributes whose names correspond with the names of the arguments. For example, a datatype which accepts a dictionary of lookup values may publish this as a sorted series of tuples. Given a previously un-cacheable type as:

class LookupType(UserDefinedType):
    """a custom type that accepts a dictionary as a parameter.

    this is the non-cacheable version, as "self.lookup" is not
    hashable.

    """

    def __init__(self, lookup):
        self.lookup = lookup

    def get_col_spec(self, **kw):
        return "VARCHAR(255)"

    def bind_processor(self, dialect): ...  # works with "self.lookup" ...
Where “lookup” is a dictionary. The type will not be able to generate a cache key:

>>> type_ = LookupType({"a": 10, "b": 20})
>>> type_._static_cache_key
<stdin>:1: SAWarning: UserDefinedType LookupType({'a': 10, 'b': 20}) will not
produce a cache key because the ``cache_ok`` flag is not set to True.
Set this flag to True if this type object's state is safe to use
in a cache key, or False to disable this warning.
symbol('no_cache')
If we did set up such a cache key, it wouldn’t be usable. We would get a tuple structure that contains a dictionary inside of it, which cannot itself be used as a key in a “cache dictionary” such as SQLAlchemy’s statement cache, since Python dictionaries aren’t hashable:

>>> # set cache_ok = True
>>> type_.cache_ok = True

>>> # this is the cache key it would generate
>>> key = type_._static_cache_key
>>> key
(<class '__main__.LookupType'>, ('lookup', {'a': 10, 'b': 20}))

>>> # however this key is not hashable, will fail when used with
>>> # SQLAlchemy statement cache
>>> some_cache = {key: "some sql value"}
Traceback (most recent call last): File "<stdin>", line 1,
in <module> TypeError: unhashable type: 'dict'
The type may be made cacheable by assigning a sorted tuple of tuples to the “.lookup” attribute:

class LookupType(UserDefinedType):
    """a custom type that accepts a dictionary as a parameter.

    The dictionary is stored both as itself in a private variable,
    and published in a public variable as a sorted tuple of tuples,
    which is hashable and will also return the same value for any
    two equivalent dictionaries.  Note it assumes the keys and
    values of the dictionary are themselves hashable.

    """

    cache_ok = True

    def __init__(self, lookup):
        self._lookup = lookup

        # assume keys/values of "lookup" are hashable; otherwise
        # they would also need to be converted in some way here
        self.lookup = tuple((key, lookup[key]) for key in sorted(lookup))

    def get_col_spec(self, **kw):
        return "VARCHAR(255)"

    def bind_processor(self, dialect): ...  # works with "self._lookup" ...
Where above, the cache key for LookupType({"a": 10, "b": 20}) will be:

>>> LookupType({"a": 10, "b": 20})._static_cache_key
(<class '__main__.LookupType'>, ('lookup', (('a', 10), ('b', 20))))
New in version 1.4.14: - added the cache_ok flag to allow some configurability of caching for TypeDecorator classes.

New in version 1.4.28: - added the ExternalType mixin which generalizes the cache_ok flag to both the TypeDecorator and UserDefinedType classes.

See also

SQL Compilation Caching

method sqlalchemy.types.Interval.coerce_compared_value(op, value)
Suggest a type for a ‘coerced’ Python value in an expression.

Given an operator and value, gives the type a chance to return a type which the value should be coerced into.

The default behavior here is conservative; if the right-hand side is already coerced into a SQL type based on its Python type, it is usually left alone.

End-user functionality extension here should generally be via TypeDecorator, which provides more liberal behavior in that it defaults to coercing the other side of the expression into this type, thus applying special Python conversions above and beyond those needed by the DBAPI to both ides. It also provides the public method TypeDecorator.coerce_compared_value() which is intended for end-user customization of this behavior.

attribute sqlalchemy.types.Interval.comparator_factory
alias of Comparator

attribute sqlalchemy.types.Interval.impl
alias of DateTime

attribute sqlalchemy.types.Interval.python_type
method sqlalchemy.types.Interval.result_processor(dialect: Dialect, coltype: Any) → _ResultProcessorType[dt.timedelta]
Return a conversion function for processing result row values.

Returns a callable which will receive a result row column value as the sole positional argument and will return a value to return to the user.

If processing is not necessary, the method should return None.

Note

This method is only called relative to a dialect specific type object, which is often private to a dialect in use and is not the same type object as the public facing one, which means it’s not feasible to subclass a TypeEngine class in order to provide an alternate TypeEngine.result_processor() method, unless subclassing the UserDefinedType class explicitly.

To provide alternate behavior for TypeEngine.result_processor(), implement a TypeDecorator class and provide an implementation of TypeDecorator.process_result_value().

See also

Augmenting Existing Types

Parameters:
dialect – Dialect instance in use.

coltype – DBAPI coltype argument received in cursor.description.

class sqlalchemy.types.LargeBinary
A type for large binary byte data.

The LargeBinary type corresponds to a large and/or unlengthed binary type for the target platform, such as BLOB on MySQL and BYTEA for PostgreSQL. It also handles the necessary conversions for the DBAPI.

Members

__init__()

Class signature

class sqlalchemy.types.LargeBinary (sqlalchemy.types._Binary)

method sqlalchemy.types.LargeBinary.__init__(length: int | None = None)
Construct a LargeBinary type.

Parameters:
length – optional, a length for the column for use in DDL statements, for those binary types that accept a length, such as the MySQL BLOB type.

class sqlalchemy.types.MatchType
Refers to the return type of the MATCH operator.

As the ColumnOperators.match() is probably the most open-ended operator in generic SQLAlchemy Core, we can’t assume the return type at SQL evaluation time, as MySQL returns a floating point, not a boolean, and other backends might do something different. So this type acts as a placeholder, currently subclassing Boolean. The type allows dialects to inject result-processing functionality if needed, and on MySQL will return floating-point values.

Class signature

class sqlalchemy.types.MatchType (sqlalchemy.types.Boolean)

class sqlalchemy.types.Numeric
Base for non-integer numeric types, such as NUMERIC, FLOAT, DECIMAL, and other variants.

The Numeric datatype when used directly will render DDL corresponding to precision numerics if available, such as NUMERIC(precision, scale). The Float subclass will attempt to render a floating-point datatype such as FLOAT(precision).

Numeric returns Python decimal.Decimal objects by default, based on the default value of True for the Numeric.asdecimal parameter. If this parameter is set to False, returned values are coerced to Python float objects.

The Float subtype, being more specific to floating point, defaults the Float.asdecimal flag to False so that the default Python datatype is float.

Note

When using a Numeric datatype against a database type that returns Python floating point values to the driver, the accuracy of the decimal conversion indicated by Numeric.asdecimal may be limited. The behavior of specific numeric/floating point datatypes is a product of the SQL datatype in use, the Python DBAPI in use, as well as strategies that may be present within the SQLAlchemy dialect in use. Users requiring specific precision/ scale are encouraged to experiment with the available datatypes in order to determine the best results.

Members

__init__(), bind_processor(), get_dbapi_type(), literal_processor(), python_type, result_processor()

Class signature

class sqlalchemy.types.Numeric (sqlalchemy.types.HasExpressionLookup, sqlalchemy.types.TypeEngine)

method sqlalchemy.types.Numeric.__init__(precision: int | None = None, scale: int | None = None, decimal_return_scale: int | None = None, asdecimal: bool = True)
Construct a Numeric.

Parameters:
precision – the numeric precision for use in DDL CREATE TABLE.

scale – the numeric scale for use in DDL CREATE TABLE.

asdecimal – default True. Return whether or not values should be sent as Python Decimal objects, or as floats. Different DBAPIs send one or the other based on datatypes - the Numeric type will ensure that return values are one or the other across DBAPIs consistently.

decimal_return_scale – Default scale to use when converting from floats to Python decimals. Floating point values will typically be much longer due to decimal inaccuracy, and most floating point database types don’t have a notion of “scale”, so by default the float type looks for the first ten decimal places when converting. Specifying this value will override that length. Types which do include an explicit “.scale” value, such as the base Numeric as well as the MySQL float types, will use the value of “.scale” as the default for decimal_return_scale, if not otherwise specified.

When using the Numeric type, care should be taken to ensure that the asdecimal setting is appropriate for the DBAPI in use - when Numeric applies a conversion from Decimal->float or float-> Decimal, this conversion incurs an additional performance overhead for all result columns received.

DBAPIs that return Decimal natively (e.g. psycopg2) will have better accuracy and higher performance with a setting of True, as the native translation to Decimal reduces the amount of floating- point issues at play, and the Numeric type itself doesn’t need to apply any further conversions. However, another DBAPI which returns floats natively will incur an additional conversion overhead, and is still subject to floating point data loss - in which case asdecimal=False will at least remove the extra conversion overhead.

method sqlalchemy.types.Numeric.bind_processor(dialect)
Return a conversion function for processing bind values.

Returns a callable which will receive a bind parameter value as the sole positional argument and will return a value to send to the DB-API.

If processing is not necessary, the method should return None.

Note

This method is only called relative to a dialect specific type object, which is often private to a dialect in use and is not the same type object as the public facing one, which means it’s not feasible to subclass a TypeEngine class in order to provide an alternate TypeEngine.bind_processor() method, unless subclassing the UserDefinedType class explicitly.

To provide alternate behavior for TypeEngine.bind_processor(), implement a TypeDecorator class and provide an implementation of TypeDecorator.process_bind_param().

See also

Augmenting Existing Types

Parameters:
dialect – Dialect instance in use.

method sqlalchemy.types.Numeric.get_dbapi_type(dbapi)
Return the corresponding type object from the underlying DB-API, if any.

This can be useful for calling setinputsizes(), for example.

method sqlalchemy.types.Numeric.literal_processor(dialect)
Return a conversion function for processing literal values that are to be rendered directly without using binds.

This function is used when the compiler makes use of the “literal_binds” flag, typically used in DDL generation as well as in certain scenarios where backends don’t accept bound parameters.

Returns a callable which will receive a literal Python value as the sole positional argument and will return a string representation to be rendered in a SQL statement.

Note

This method is only called relative to a dialect specific type object, which is often private to a dialect in use and is not the same type object as the public facing one, which means it’s not feasible to subclass a TypeEngine class in order to provide an alternate TypeEngine.literal_processor() method, unless subclassing the UserDefinedType class explicitly.

To provide alternate behavior for TypeEngine.literal_processor(), implement a TypeDecorator class and provide an implementation of TypeDecorator.process_literal_param().

See also

Augmenting Existing Types

attribute sqlalchemy.types.Numeric.python_type
method sqlalchemy.types.Numeric.result_processor(dialect, coltype)
Return a conversion function for processing result row values.

Returns a callable which will receive a result row column value as the sole positional argument and will return a value to return to the user.

If processing is not necessary, the method should return None.

Note

This method is only called relative to a dialect specific type object, which is often private to a dialect in use and is not the same type object as the public facing one, which means it’s not feasible to subclass a TypeEngine class in order to provide an alternate TypeEngine.result_processor() method, unless subclassing the UserDefinedType class explicitly.

To provide alternate behavior for TypeEngine.result_processor(), implement a TypeDecorator class and provide an implementation of TypeDecorator.process_result_value().

See also

Augmenting Existing Types

Parameters:
dialect – Dialect instance in use.

coltype – DBAPI coltype argument received in cursor.description.

class sqlalchemy.types.PickleType
Holds Python objects, which are serialized using pickle.

PickleType builds upon the Binary type to apply Python’s pickle.dumps() to incoming objects, and pickle.loads() on the way out, allowing any pickleable Python object to be stored as a serialized binary field.

To allow ORM change events to propagate for elements associated with PickleType, see Mutation Tracking.

Members

__init__(), bind_processor(), cache_ok, compare_values(), impl, result_processor()

Class signature

class sqlalchemy.types.PickleType (sqlalchemy.types.TypeDecorator)

method sqlalchemy.types.PickleType.__init__(protocol: int = 5, pickler: Any = None, comparator: Callable[[Any, Any], bool] | None = None, impl: _TypeEngineArgument[Any] | None = None)
Construct a PickleType.

Parameters:
protocol – defaults to pickle.HIGHEST_PROTOCOL.

pickler – defaults to pickle. May be any object with pickle-compatible dumps and loads methods.

comparator – a 2-arg callable predicate used to compare values of this type. If left as None, the Python “equals” operator is used to compare values.

impl –

A binary-storing TypeEngine class or instance to use in place of the default LargeBinary. For example the :class: _mysql.LONGBLOB class may be more effective when using MySQL.

New in version 1.4.20.

method sqlalchemy.types.PickleType.bind_processor(dialect)
Provide a bound value processing function for the given Dialect.

This is the method that fulfills the TypeEngine contract for bound value conversion which normally occurs via the TypeEngine.bind_processor() method.

Note

User-defined subclasses of TypeDecorator should not implement this method, and should instead implement TypeDecorator.process_bind_param() so that the “inner” processing provided by the implementing type is maintained.

Parameters:
dialect – Dialect instance in use.

attribute sqlalchemy.types.PickleType.cache_ok: bool | None = True
Indicate if statements using this ExternalType are “safe to cache”.

The default value None will emit a warning and then not allow caching of a statement which includes this type. Set to False to disable statements using this type from being cached at all without a warning. When set to True, the object’s class and selected elements from its state will be used as part of the cache key. For example, using a TypeDecorator:

class MyType(TypeDecorator):
    impl = String

    cache_ok = True

    def __init__(self, choices):
        self.choices = tuple(choices)
        self.internal_only = True
The cache key for the above type would be equivalent to:

>>> MyType(["a", "b", "c"])._static_cache_key
(<class '__main__.MyType'>, ('choices', ('a', 'b', 'c')))
The caching scheme will extract attributes from the type that correspond to the names of parameters in the __init__() method. Above, the “choices” attribute becomes part of the cache key but “internal_only” does not, because there is no parameter named “internal_only”.

The requirements for cacheable elements is that they are hashable and also that they indicate the same SQL rendered for expressions using this type every time for a given cache value.

To accommodate for datatypes that refer to unhashable structures such as dictionaries, sets and lists, these objects can be made “cacheable” by assigning hashable structures to the attributes whose names correspond with the names of the arguments. For example, a datatype which accepts a dictionary of lookup values may publish this as a sorted series of tuples. Given a previously un-cacheable type as:

class LookupType(UserDefinedType):
    """a custom type that accepts a dictionary as a parameter.

    this is the non-cacheable version, as "self.lookup" is not
    hashable.

    """

    def __init__(self, lookup):
        self.lookup = lookup

    def get_col_spec(self, **kw):
        return "VARCHAR(255)"

    def bind_processor(self, dialect): ...  # works with "self.lookup" ...
Where “lookup” is a dictionary. The type will not be able to generate a cache key:

>>> type_ = LookupType({"a": 10, "b": 20})
>>> type_._static_cache_key
<stdin>:1: SAWarning: UserDefinedType LookupType({'a': 10, 'b': 20}) will not
produce a cache key because the ``cache_ok`` flag is not set to True.
Set this flag to True if this type object's state is safe to use
in a cache key, or False to disable this warning.
symbol('no_cache')
If we did set up such a cache key, it wouldn’t be usable. We would get a tuple structure that contains a dictionary inside of it, which cannot itself be used as a key in a “cache dictionary” such as SQLAlchemy’s statement cache, since Python dictionaries aren’t hashable:

>>> # set cache_ok = True
>>> type_.cache_ok = True

>>> # this is the cache key it would generate
>>> key = type_._static_cache_key
>>> key
(<class '__main__.LookupType'>, ('lookup', {'a': 10, 'b': 20}))

>>> # however this key is not hashable, will fail when used with
>>> # SQLAlchemy statement cache
>>> some_cache = {key: "some sql value"}
Traceback (most recent call last): File "<stdin>", line 1,
in <module> TypeError: unhashable type: 'dict'
The type may be made cacheable by assigning a sorted tuple of tuples to the “.lookup” attribute:

class LookupType(UserDefinedType):
    """a custom type that accepts a dictionary as a parameter.

    The dictionary is stored both as itself in a private variable,
    and published in a public variable as a sorted tuple of tuples,
    which is hashable and will also return the same value for any
    two equivalent dictionaries.  Note it assumes the keys and
    values of the dictionary are themselves hashable.

    """

    cache_ok = True

    def __init__(self, lookup):
        self._lookup = lookup

        # assume keys/values of "lookup" are hashable; otherwise
        # they would also need to be converted in some way here
        self.lookup = tuple((key, lookup[key]) for key in sorted(lookup))

    def get_col_spec(self, **kw):
        return "VARCHAR(255)"

    def bind_processor(self, dialect): ...  # works with "self._lookup" ...
Where above, the cache key for LookupType({"a": 10, "b": 20}) will be:

>>> LookupType({"a": 10, "b": 20})._static_cache_key
(<class '__main__.LookupType'>, ('lookup', (('a', 10), ('b', 20))))
New in version 1.4.14: - added the cache_ok flag to allow some configurability of caching for TypeDecorator classes.

New in version 1.4.28: - added the ExternalType mixin which generalizes the cache_ok flag to both the TypeDecorator and UserDefinedType classes.

See also

SQL Compilation Caching

method sqlalchemy.types.PickleType.compare_values(x, y)
Given two values, compare them for equality.

By default this calls upon TypeEngine.compare_values() of the underlying “impl”, which in turn usually uses the Python equals operator ==.

This function is used by the ORM to compare an original-loaded value with an intercepted “changed” value, to determine if a net change has occurred.

attribute sqlalchemy.types.PickleType.impl
alias of LargeBinary

method sqlalchemy.types.PickleType.result_processor(dialect, coltype)
Provide a result value processing function for the given Dialect.

This is the method that fulfills the TypeEngine contract for bound value conversion which normally occurs via the TypeEngine.result_processor() method.

Note

User-defined subclasses of TypeDecorator should not implement this method, and should instead implement TypeDecorator.process_result_value() so that the “inner” processing provided by the implementing type is maintained.

Parameters:
dialect – Dialect instance in use.

coltype – A SQLAlchemy data type

class sqlalchemy.types.SchemaType
Add capabilities to a type which allow for schema-level DDL to be associated with a type.

Supports types that must be explicitly created/dropped (i.e. PG ENUM type) as well as types that are complimented by table or schema level constraints, triggers, and other rules.

SchemaType classes can also be targets for the DDLEvents.before_parent_attach() and DDLEvents.after_parent_attach() events, where the events fire off surrounding the association of the type object with a parent Column.

See also

Enum

Boolean

Members

adapt(), copy(), create(), drop(), name

Class signature

class sqlalchemy.types.SchemaType (sqlalchemy.sql.expression.SchemaEventTarget, sqlalchemy.types.TypeEngineMixin)

method sqlalchemy.types.SchemaType.adapt(cls: Type[TypeEngine | TypeEngineMixin], **kw: Any) → TypeEngine
method sqlalchemy.types.SchemaType.copy(**kw)
method sqlalchemy.types.SchemaType.create(bind, checkfirst=False)
Issue CREATE DDL for this type, if applicable.

method sqlalchemy.types.SchemaType.drop(bind, checkfirst=False)
Issue DROP DDL for this type, if applicable.

attribute sqlalchemy.types.SchemaType.name: str | None
class sqlalchemy.types.SmallInteger
A type for smaller int integers.

Typically generates a SMALLINT in DDL, and otherwise acts like a normal Integer on the Python side.

Class signature

class sqlalchemy.types.SmallInteger (sqlalchemy.types.Integer)

class sqlalchemy.types.String
The base for all string and character types.

In SQL, corresponds to VARCHAR.

The length field is usually required when the String type is used within a CREATE TABLE statement, as VARCHAR requires a length on most databases.

Members

__init__(), bind_processor(), get_dbapi_type(), literal_processor(), python_type, result_processor()

Class signature

class sqlalchemy.types.String (sqlalchemy.types.Concatenable, sqlalchemy.types.TypeEngine)

method sqlalchemy.types.String.__init__(length: int | None = None, collation: str | None = None)
Create a string-holding type.

Parameters:
length – optional, a length for the column for use in DDL and CAST expressions. May be safely omitted if no CREATE TABLE will be issued. Certain databases may require a length for use in DDL, and will raise an exception when the CREATE TABLE DDL is issued if a VARCHAR with no length is included. Whether the value is interpreted as bytes or characters is database specific.

collation –

Optional, a column-level collation for use in DDL and CAST expressions. Renders using the COLLATE keyword supported by SQLite, MySQL, and PostgreSQL. E.g.:

>>> from sqlalchemy import cast, select, String
>>> print(select(cast("some string", String(collation="utf8"))))
SELECT CAST(:param_1 AS VARCHAR COLLATE utf8) AS anon_1
Note

In most cases, the Unicode or UnicodeText datatypes should be used for a Column that expects to store non-ascii data. These datatypes will ensure that the correct types are used on the database.

method sqlalchemy.types.String.bind_processor(dialect)
Return a conversion function for processing bind values.

Returns a callable which will receive a bind parameter value as the sole positional argument and will return a value to send to the DB-API.

If processing is not necessary, the method should return None.

Note

This method is only called relative to a dialect specific type object, which is often private to a dialect in use and is not the same type object as the public facing one, which means it’s not feasible to subclass a TypeEngine class in order to provide an alternate TypeEngine.bind_processor() method, unless subclassing the UserDefinedType class explicitly.

To provide alternate behavior for TypeEngine.bind_processor(), implement a TypeDecorator class and provide an implementation of TypeDecorator.process_bind_param().

See also

Augmenting Existing Types

Parameters:
dialect – Dialect instance in use.

method sqlalchemy.types.String.get_dbapi_type(dbapi)
Return the corresponding type object from the underlying DB-API, if any.

This can be useful for calling setinputsizes(), for example.

method sqlalchemy.types.String.literal_processor(dialect)
Return a conversion function for processing literal values that are to be rendered directly without using binds.

This function is used when the compiler makes use of the “literal_binds” flag, typically used in DDL generation as well as in certain scenarios where backends don’t accept bound parameters.

Returns a callable which will receive a literal Python value as the sole positional argument and will return a string representation to be rendered in a SQL statement.

Note

This method is only called relative to a dialect specific type object, which is often private to a dialect in use and is not the same type object as the public facing one, which means it’s not feasible to subclass a TypeEngine class in order to provide an alternate TypeEngine.literal_processor() method, unless subclassing the UserDefinedType class explicitly.

To provide alternate behavior for TypeEngine.literal_processor(), implement a TypeDecorator class and provide an implementation of TypeDecorator.process_literal_param().

See also

Augmenting Existing Types

attribute sqlalchemy.types.String.python_type
method sqlalchemy.types.String.result_processor(dialect, coltype)
Return a conversion function for processing result row values.

Returns a callable which will receive a result row column value as the sole positional argument and will return a value to return to the user.

If processing is not necessary, the method should return None.

Note

This method is only called relative to a dialect specific type object, which is often private to a dialect in use and is not the same type object as the public facing one, which means it’s not feasible to subclass a TypeEngine class in order to provide an alternate TypeEngine.result_processor() method, unless subclassing the UserDefinedType class explicitly.

To provide alternate behavior for TypeEngine.result_processor(), implement a TypeDecorator class and provide an implementation of TypeDecorator.process_result_value().

See also

Augmenting Existing Types

Parameters:
dialect – Dialect instance in use.

coltype – DBAPI coltype argument received in cursor.description.

class sqlalchemy.types.Text
A variably sized string type.

In SQL, usually corresponds to CLOB or TEXT. In general, TEXT objects do not have a length; while some databases will accept a length argument here, it will be rejected by others.

Class signature

class sqlalchemy.types.Text (sqlalchemy.types.String)

class sqlalchemy.types.Time
A type for datetime.time() objects.

Members

get_dbapi_type(), literal_processor(), python_type

Class signature

class sqlalchemy.types.Time (sqlalchemy.types._RenderISO8601NoT, sqlalchemy.types.HasExpressionLookup, sqlalchemy.types.TypeEngine)

method sqlalchemy.types.Time.get_dbapi_type(dbapi)
Return the corresponding type object from the underlying DB-API, if any.

This can be useful for calling setinputsizes(), for example.

method sqlalchemy.types.Time.literal_processor(dialect)
Return a conversion function for processing literal values that are to be rendered directly without using binds.

This function is used when the compiler makes use of the “literal_binds” flag, typically used in DDL generation as well as in certain scenarios where backends don’t accept bound parameters.

Returns a callable which will receive a literal Python value as the sole positional argument and will return a string representation to be rendered in a SQL statement.

Note

This method is only called relative to a dialect specific type object, which is often private to a dialect in use and is not the same type object as the public facing one, which means it’s not feasible to subclass a TypeEngine class in order to provide an alternate TypeEngine.literal_processor() method, unless subclassing the UserDefinedType class explicitly.

To provide alternate behavior for TypeEngine.literal_processor(), implement a TypeDecorator class and provide an implementation of TypeDecorator.process_literal_param().

See also

Augmenting Existing Types

attribute sqlalchemy.types.Time.python_type
class sqlalchemy.types.Unicode
A variable length Unicode string type.

The Unicode type is a String subclass that assumes input and output strings that may contain non-ASCII characters, and for some backends implies an underlying column type that is explicitly supporting of non-ASCII data, such as NVARCHAR on Oracle Database and SQL Server. This will impact the output of CREATE TABLE statements and CAST functions at the dialect level.

The character encoding used by the Unicode type that is used to transmit and receive data to the database is usually determined by the DBAPI itself. All modern DBAPIs accommodate non-ASCII strings but may have different methods of managing database encodings; if necessary, this encoding should be configured as detailed in the notes for the target DBAPI in the Dialects section.

In modern SQLAlchemy, use of the Unicode datatype does not imply any encoding/decoding behavior within SQLAlchemy itself. In Python 3, all string objects are inherently Unicode capable, and SQLAlchemy does not produce bytestring objects nor does it accommodate a DBAPI that does not return Python Unicode objects in result sets for string values.

Warning

Some database backends, particularly SQL Server with pyodbc, are known to have undesirable behaviors regarding data that is noted as being of NVARCHAR type as opposed to VARCHAR, including datatype mismatch errors and non-use of indexes. See the section on DialectEvents.do_setinputsizes() for background on working around unicode character issues for backends like SQL Server with pyodbc as well as cx_Oracle.

See also

UnicodeText - unlengthed textual counterpart to Unicode.

DialectEvents.do_setinputsizes()

Class signature

class sqlalchemy.types.Unicode (sqlalchemy.types.String)

class sqlalchemy.types.UnicodeText
An unbounded-length Unicode string type.

See Unicode for details on the unicode behavior of this object.

Like Unicode, usage the UnicodeText type implies a unicode-capable type being used on the backend, such as NCLOB, NTEXT.

Class signature

class sqlalchemy.types.UnicodeText (sqlalchemy.types.Text)

class sqlalchemy.types.Uuid
Represent a database agnostic UUID datatype.

For backends that have no “native” UUID datatype, the value will make use of CHAR(32) and store the UUID as a 32-character alphanumeric hex string.

For backends which are known to support UUID directly or a similar uuid-storing datatype such as SQL Server’s UNIQUEIDENTIFIER, a “native” mode enabled by default allows these types will be used on those backends.

In its default mode of use, the Uuid datatype expects Python uuid objects, from the Python uuid module:

import uuid

from sqlalchemy import Uuid
from sqlalchemy import Table, Column, MetaData, String


metadata_obj = MetaData()

t = Table(
    "t",
    metadata_obj,
    Column("uuid_data", Uuid, primary_key=True),
    Column("other_data", String),
)

with engine.begin() as conn:
    conn.execute(
        t.insert(), {"uuid_data": uuid.uuid4(), "other_data": "some data"}
    )
To have the Uuid datatype work with string-based Uuids (e.g. 32 character hexadecimal strings), pass the Uuid.as_uuid parameter with the value False.

New in version 2.0.

See also

UUID - represents exactly the UUID datatype without any backend-agnostic behaviors.

Members

__init__(), bind_processor(), coerce_compared_value(), literal_processor(), python_type, result_processor()

Class signature

class sqlalchemy.types.Uuid (sqlalchemy.types.Emulated, sqlalchemy.types.TypeEngine)

method sqlalchemy.types.Uuid.__init__(as_uuid: bool = True, native_uuid: bool = True)
Construct a Uuid type.

Parameters:
as_uuid=True –

if True, values will be interpreted as Python uuid objects, converting to/from string via the DBAPI.

native_uuid=True – if True, backends that support either the UUID datatype directly, or a UUID-storing value (such as SQL Server’s UNIQUEIDENTIFIER will be used by those backends. If False, a CHAR(32) datatype will be used for all backends regardless of native support.

method sqlalchemy.types.Uuid.bind_processor(dialect)
Return a conversion function for processing bind values.

Returns a callable which will receive a bind parameter value as the sole positional argument and will return a value to send to the DB-API.

If processing is not necessary, the method should return None.

Note

This method is only called relative to a dialect specific type object, which is often private to a dialect in use and is not the same type object as the public facing one, which means it’s not feasible to subclass a TypeEngine class in order to provide an alternate TypeEngine.bind_processor() method, unless subclassing the UserDefinedType class explicitly.

To provide alternate behavior for TypeEngine.bind_processor(), implement a TypeDecorator class and provide an implementation of TypeDecorator.process_bind_param().

See also

Augmenting Existing Types

Parameters:
dialect – Dialect instance in use.

method sqlalchemy.types.Uuid.coerce_compared_value(op, value)
See TypeEngine.coerce_compared_value() for a description.

method sqlalchemy.types.Uuid.literal_processor(dialect)
Return a conversion function for processing literal values that are to be rendered directly without using binds.

This function is used when the compiler makes use of the “literal_binds” flag, typically used in DDL generation as well as in certain scenarios where backends don’t accept bound parameters.

Returns a callable which will receive a literal Python value as the sole positional argument and will return a string representation to be rendered in a SQL statement.

Note

This method is only called relative to a dialect specific type object, which is often private to a dialect in use and is not the same type object as the public facing one, which means it’s not feasible to subclass a TypeEngine class in order to provide an alternate TypeEngine.literal_processor() method, unless subclassing the UserDefinedType class explicitly.

To provide alternate behavior for TypeEngine.literal_processor(), implement a TypeDecorator class and provide an implementation of TypeDecorator.process_literal_param().

See also

Augmenting Existing Types

attribute sqlalchemy.types.Uuid.python_type
method sqlalchemy.types.Uuid.result_processor(dialect, coltype)
Return a conversion function for processing result row values.

Returns a callable which will receive a result row column value as the sole positional argument and will return a value to return to the user.

If processing is not necessary, the method should return None.

Note

This method is only called relative to a dialect specific type object, which is often private to a dialect in use and is not the same type object as the public facing one, which means it’s not feasible to subclass a TypeEngine class in order to provide an alternate TypeEngine.result_processor() method, unless subclassing the UserDefinedType class explicitly.

To provide alternate behavior for TypeEngine.result_processor(), implement a TypeDecorator class and provide an implementation of TypeDecorator.process_result_value().

See also

Augmenting Existing Types

Parameters:
dialect – Dialect instance in use.

coltype – DBAPI coltype argument received in cursor.description.

SQL Standard and Multiple Vendor “UPPERCASE” Types
This category of types refers to types that are either part of the SQL standard, or are potentially found within a subset of database backends. Unlike the “generic” types, the SQL standard/multi-vendor types have no guarantee of working on all backends, and will only work on those backends that explicitly support them by name. That is, the type will always emit its exact name in DDL with CREATE TABLE is issued.

Object Name	Description
ARRAY

Represent a SQL Array type.

BIGINT

The SQL BIGINT type.

BINARY

The SQL BINARY type.

BLOB

The SQL BLOB type.

BOOLEAN

The SQL BOOLEAN type.

CHAR

The SQL CHAR type.

CLOB

The CLOB type.

DATE

The SQL DATE type.

DATETIME

The SQL DATETIME type.

DECIMAL

The SQL DECIMAL type.

DOUBLE

The SQL DOUBLE type.

DOUBLE_PRECISION

The SQL DOUBLE PRECISION type.

FLOAT

The SQL FLOAT type.

INT

alias of INTEGER

INTEGER

The SQL INT or INTEGER type.

JSON

Represent a SQL JSON type.

NCHAR

The SQL NCHAR type.

NUMERIC

The SQL NUMERIC type.

NVARCHAR

The SQL NVARCHAR type.

REAL

The SQL REAL type.

SMALLINT

The SQL SMALLINT type.

TEXT

The SQL TEXT type.

TIME

The SQL TIME type.

TIMESTAMP

The SQL TIMESTAMP type.

UUID

Represent the SQL UUID type.

VARBINARY

The SQL VARBINARY type.

VARCHAR

The SQL VARCHAR type.

class sqlalchemy.types.ARRAY
Represent a SQL Array type.

Note

This type serves as the basis for all ARRAY operations. However, currently only the PostgreSQL backend has support for SQL arrays in SQLAlchemy. It is recommended to use the PostgreSQL-specific sqlalchemy.dialects.postgresql.ARRAY type directly when using ARRAY types with PostgreSQL, as it provides additional operators specific to that backend.

ARRAY is part of the Core in support of various SQL standard functions such as array_agg which explicitly involve arrays; however, with the exception of the PostgreSQL backend and possibly some third-party dialects, no other SQLAlchemy built-in dialect has support for this type.

An ARRAY type is constructed given the “type” of element:

mytable = Table("mytable", metadata, Column("data", ARRAY(Integer)))
The above type represents an N-dimensional array, meaning a supporting backend such as PostgreSQL will interpret values with any number of dimensions automatically. To produce an INSERT construct that passes in a 1-dimensional array of integers:

connection.execute(mytable.insert(), {"data": [1, 2, 3]})
The ARRAY type can be constructed given a fixed number of dimensions:

mytable = Table(
    "mytable", metadata, Column("data", ARRAY(Integer, dimensions=2))
)
Sending a number of dimensions is optional, but recommended if the datatype is to represent arrays of more than one dimension. This number is used:

When emitting the type declaration itself to the database, e.g. INTEGER[][]

When translating Python values to database values, and vice versa, e.g. an ARRAY of Unicode objects uses this number to efficiently access the string values inside of array structures without resorting to per-row type inspection

When used with the Python getitem accessor, the number of dimensions serves to define the kind of type that the [] operator should return, e.g. for an ARRAY of INTEGER with two dimensions:

>>> expr = table.c.column[5]  # returns ARRAY(Integer, dimensions=1)
>>> expr = expr[6]  # returns Integer
For 1-dimensional arrays, an ARRAY instance with no dimension parameter will generally assume single-dimensional behaviors.

SQL expressions of type ARRAY have support for “index” and “slice” behavior. The [] operator produces expression constructs which will produce the appropriate SQL, both for SELECT statements:

select(mytable.c.data[5], mytable.c.data[2:7])
as well as UPDATE statements when the Update.values() method is used:

mytable.update().values(
    {mytable.c.data[5]: 7, mytable.c.data[2:7]: [1, 2, 3]}
)
Indexed access is one-based by default; for zero-based index conversion, set ARRAY.zero_indexes.

The ARRAY type also provides for the operators Comparator.any() and Comparator.all(). The PostgreSQL-specific version of ARRAY also provides additional operators.

Detecting Changes in ARRAY columns when using the ORM

The ARRAY type, when used with the SQLAlchemy ORM, does not detect in-place mutations to the array. In order to detect these, the sqlalchemy.ext.mutable extension must be used, using the MutableList class:

from sqlalchemy import ARRAY
from sqlalchemy.ext.mutable import MutableList


class SomeOrmClass(Base):
    # ...

    data = Column(MutableList.as_mutable(ARRAY(Integer)))
This extension will allow “in-place” changes such to the array such as .append() to produce events which will be detected by the unit of work. Note that changes to elements inside the array, including subarrays that are mutated in place, are not detected.

Alternatively, assigning a new array value to an ORM element that replaces the old one will always trigger a change event.

See also

sqlalchemy.dialects.postgresql.ARRAY

Members

__init__(), contains(), any(), all()

Class signature

class sqlalchemy.types.ARRAY (sqlalchemy.sql.expression.SchemaEventTarget, sqlalchemy.types.Indexable, sqlalchemy.types.Concatenable, sqlalchemy.types.TypeEngine)

method sqlalchemy.types.ARRAY.__init__(item_type: _TypeEngineArgument[Any], as_tuple: bool = False, dimensions: int | None = None, zero_indexes: bool = False)
Construct an ARRAY.

E.g.:

Column("myarray", ARRAY(Integer))
Arguments are:

Parameters:
item_type – The data type of items of this array. Note that dimensionality is irrelevant here, so multi-dimensional arrays like INTEGER[][], are constructed as ARRAY(Integer), not as ARRAY(ARRAY(Integer)) or such.

as_tuple=False – Specify whether return results should be converted to tuples from lists. This parameter is not generally needed as a Python list corresponds well to a SQL array.

dimensions – if non-None, the ARRAY will assume a fixed number of dimensions. This impacts how the array is declared on the database, how it goes about interpreting Python and result values, as well as how expression behavior in conjunction with the “getitem” operator works. See the description at ARRAY for additional detail.

zero_indexes=False – when True, index values will be converted between Python zero-based and SQL one-based indexes, e.g. a value of one will be added to all index values before passing to the database.

class Comparator
Define comparison operations for ARRAY.

More operators are available on the dialect-specific form of this type. See Comparator.

Class signature

class sqlalchemy.types.ARRAY.Comparator (sqlalchemy.types.Comparator, sqlalchemy.types.Comparator)

method sqlalchemy.types.ARRAY.Comparator.contains(*arg, **kw)
ARRAY.contains() not implemented for the base ARRAY type. Use the dialect-specific ARRAY type.

See also

ARRAY - PostgreSQL specific version.

method sqlalchemy.types.ARRAY.Comparator.any(other, operator=None)
Return other operator ANY (array) clause.

Legacy Feature

This method is an ARRAY - specific construct that is now superseded by the any_() function, which features a different calling style. The any_() function is also mirrored at the method level via the ColumnOperators.any_() method.

Usage of array-specific Comparator.any() is as follows:

from sqlalchemy.sql import operators

conn.execute(
    select(table.c.data).where(table.c.data.any(7, operator=operators.lt))
)
Parameters:
other – expression to be compared

operator – an operator object from the sqlalchemy.sql.operators package, defaults to eq().

See also

any_()

Comparator.all()

method sqlalchemy.types.ARRAY.Comparator.all(other, operator=None)
Return other operator ALL (array) clause.

Legacy Feature

This method is an ARRAY - specific construct that is now superseded by the all_() function, which features a different calling style. The all_() function is also mirrored at the method level via the ColumnOperators.all_() method.

Usage of array-specific Comparator.all() is as follows:

from sqlalchemy.sql import operators

conn.execute(
    select(table.c.data).where(table.c.data.all(7, operator=operators.lt))
)
Parameters:
other – expression to be compared

operator – an operator object from the sqlalchemy.sql.operators package, defaults to eq().

See also

all_()

Comparator.any()

class sqlalchemy.types.BIGINT
The SQL BIGINT type.

See also

BigInteger - documentation for the base type.

Class signature

class sqlalchemy.types.BIGINT (sqlalchemy.types.BigInteger)

class sqlalchemy.types.BINARY
The SQL BINARY type.

Class signature

class sqlalchemy.types.BINARY (sqlalchemy.types._Binary)

class sqlalchemy.types.BLOB
The SQL BLOB type.

Members

__init__()

Class signature

class sqlalchemy.types.BLOB (sqlalchemy.types.LargeBinary)

method sqlalchemy.types.BLOB.__init__(length: int | None = None)
inherited from the sqlalchemy.types.LargeBinary.__init__ method of LargeBinary

Construct a LargeBinary type.

Parameters:
length – optional, a length for the column for use in DDL statements, for those binary types that accept a length, such as the MySQL BLOB type.

class sqlalchemy.types.BOOLEAN
The SQL BOOLEAN type.

Members

__init__()

Class signature

class sqlalchemy.types.BOOLEAN (sqlalchemy.types.Boolean)

method sqlalchemy.types.BOOLEAN.__init__(create_constraint: bool = False, name: str | None = None, _create_events: bool = True, _adapted_from: SchemaType | None = None)
inherited from the sqlalchemy.types.Boolean.__init__ method of Boolean

Construct a Boolean.

Parameters:
create_constraint –

defaults to False. If the boolean is generated as an int/smallint, also create a CHECK constraint on the table that ensures 1 or 0 as a value.

Note

it is strongly recommended that the CHECK constraint have an explicit name in order to support schema-management concerns. This can be established either by setting the Boolean.name parameter or by setting up an appropriate naming convention; see Configuring Constraint Naming Conventions for background.

Changed in version 1.4: - this flag now defaults to False, meaning no CHECK constraint is generated for a non-native enumerated type.

name – if a CHECK constraint is generated, specify the name of the constraint.

class sqlalchemy.types.CHAR
The SQL CHAR type.

Members

__init__()

Class signature

class sqlalchemy.types.CHAR (sqlalchemy.types.String)

method sqlalchemy.types.CHAR.__init__(length: int | None = None, collation: str | None = None)
inherited from the sqlalchemy.types.String.__init__ method of String

Create a string-holding type.

Parameters:
length – optional, a length for the column for use in DDL and CAST expressions. May be safely omitted if no CREATE TABLE will be issued. Certain databases may require a length for use in DDL, and will raise an exception when the CREATE TABLE DDL is issued if a VARCHAR with no length is included. Whether the value is interpreted as bytes or characters is database specific.

collation –

Optional, a column-level collation for use in DDL and CAST expressions. Renders using the COLLATE keyword supported by SQLite, MySQL, and PostgreSQL. E.g.:

>>> from sqlalchemy import cast, select, String
>>> print(select(cast("some string", String(collation="utf8"))))
SELECT CAST(:param_1 AS VARCHAR COLLATE utf8) AS anon_1
Note

In most cases, the Unicode or UnicodeText datatypes should be used for a Column that expects to store non-ascii data. These datatypes will ensure that the correct types are used on the database.

class sqlalchemy.types.CLOB
The CLOB type.

This type is found in Oracle Database and Informix.

Members

__init__()

Class signature

class sqlalchemy.types.CLOB (sqlalchemy.types.Text)

method sqlalchemy.types.CLOB.__init__(length: int | None = None, collation: str | None = None)
inherited from the sqlalchemy.types.String.__init__ method of String

Create a string-holding type.

Parameters:
length – optional, a length for the column for use in DDL and CAST expressions. May be safely omitted if no CREATE TABLE will be issued. Certain databases may require a length for use in DDL, and will raise an exception when the CREATE TABLE DDL is issued if a VARCHAR with no length is included. Whether the value is interpreted as bytes or characters is database specific.

collation –

Optional, a column-level collation for use in DDL and CAST expressions. Renders using the COLLATE keyword supported by SQLite, MySQL, and PostgreSQL. E.g.:

>>> from sqlalchemy import cast, select, String
>>> print(select(cast("some string", String(collation="utf8"))))
SELECT CAST(:param_1 AS VARCHAR COLLATE utf8) AS anon_1
Note

In most cases, the Unicode or UnicodeText datatypes should be used for a Column that expects to store non-ascii data. These datatypes will ensure that the correct types are used on the database.

class sqlalchemy.types.DATE
The SQL DATE type.

Class signature

class sqlalchemy.types.DATE (sqlalchemy.types.Date)

class sqlalchemy.types.DATETIME
The SQL DATETIME type.

Members

__init__()

Class signature

class sqlalchemy.types.DATETIME (sqlalchemy.types.DateTime)

method sqlalchemy.types.DATETIME.__init__(timezone: bool = False)
inherited from the sqlalchemy.types.DateTime.__init__ method of DateTime

Construct a new DateTime.

Parameters:
timezone – boolean. Indicates that the datetime type should enable timezone support, if available on the base date/time-holding type only. It is recommended to make use of the TIMESTAMP datatype directly when using this flag, as some databases include separate generic date/time-holding types distinct from the timezone-capable TIMESTAMP datatype, such as Oracle Database.

class sqlalchemy.types.DECIMAL
The SQL DECIMAL type.

See also

Numeric - documentation for the base type.

Members

__init__()

Class signature

class sqlalchemy.types.DECIMAL (sqlalchemy.types.Numeric)

method sqlalchemy.types.DECIMAL.__init__(precision: int | None = None, scale: int | None = None, decimal_return_scale: int | None = None, asdecimal: bool = True)
inherited from the sqlalchemy.types.Numeric.__init__ method of Numeric

Construct a Numeric.

Parameters:
precision – the numeric precision for use in DDL CREATE TABLE.

scale – the numeric scale for use in DDL CREATE TABLE.

asdecimal – default True. Return whether or not values should be sent as Python Decimal objects, or as floats. Different DBAPIs send one or the other based on datatypes - the Numeric type will ensure that return values are one or the other across DBAPIs consistently.

decimal_return_scale – Default scale to use when converting from floats to Python decimals. Floating point values will typically be much longer due to decimal inaccuracy, and most floating point database types don’t have a notion of “scale”, so by default the float type looks for the first ten decimal places when converting. Specifying this value will override that length. Types which do include an explicit “.scale” value, such as the base Numeric as well as the MySQL float types, will use the value of “.scale” as the default for decimal_return_scale, if not otherwise specified.

When using the Numeric type, care should be taken to ensure that the asdecimal setting is appropriate for the DBAPI in use - when Numeric applies a conversion from Decimal->float or float-> Decimal, this conversion incurs an additional performance overhead for all result columns received.

DBAPIs that return Decimal natively (e.g. psycopg2) will have better accuracy and higher performance with a setting of True, as the native translation to Decimal reduces the amount of floating- point issues at play, and the Numeric type itself doesn’t need to apply any further conversions. However, another DBAPI which returns floats natively will incur an additional conversion overhead, and is still subject to floating point data loss - in which case asdecimal=False will at least remove the extra conversion overhead.

class sqlalchemy.types.DOUBLE
The SQL DOUBLE type.

New in version 2.0.

See also

Double - documentation for the base type.

Members

__init__()

Class signature

class sqlalchemy.types.DOUBLE (sqlalchemy.types.Double)

method sqlalchemy.types.DOUBLE.__init__(precision: int | None = None, asdecimal: bool = False, decimal_return_scale: int | None = None)
inherited from the sqlalchemy.types.Float.__init__ method of Float

Construct a Float.

Parameters:
precision –

the numeric precision for use in DDL CREATE TABLE. Backends should attempt to ensure this precision indicates a number of digits for the generic Float datatype.

Note

For the Oracle Database backend, the Float.precision parameter is not accepted when rendering DDL, as Oracle Database does not support float precision specified as a number of decimal places. Instead, use the Oracle Database-specific FLOAT datatype and specify the FLOAT.binary_precision parameter. This is new in version 2.0 of SQLAlchemy.

To create a database agnostic Float that separately specifies binary precision for Oracle Database, use TypeEngine.with_variant() as follows:

from sqlalchemy import Column
from sqlalchemy import Float
from sqlalchemy.dialects import oracle

Column(
    "float_data",
    Float(5).with_variant(oracle.FLOAT(binary_precision=16), "oracle"),
)
asdecimal – the same flag as that of Numeric, but defaults to False. Note that setting this flag to True results in floating point conversion.

decimal_return_scale – Default scale to use when converting from floats to Python decimals. Floating point values will typically be much longer due to decimal inaccuracy, and most floating point database types don’t have a notion of “scale”, so by default the float type looks for the first ten decimal places when converting. Specifying this value will override that length. Note that the MySQL float types, which do include “scale”, will use “scale” as the default for decimal_return_scale, if not otherwise specified.

class sqlalchemy.types.DOUBLE_PRECISION
The SQL DOUBLE PRECISION type.

New in version 2.0.

See also

Double - documentation for the base type.

Members

__init__()

Class signature

class sqlalchemy.types.DOUBLE_PRECISION (sqlalchemy.types.Double)

method sqlalchemy.types.DOUBLE_PRECISION.__init__(precision: int | None = None, asdecimal: bool = False, decimal_return_scale: int | None = None)
inherited from the sqlalchemy.types.Float.__init__ method of Float

Construct a Float.

Parameters:
precision –

the numeric precision for use in DDL CREATE TABLE. Backends should attempt to ensure this precision indicates a number of digits for the generic Float datatype.

Note

For the Oracle Database backend, the Float.precision parameter is not accepted when rendering DDL, as Oracle Database does not support float precision specified as a number of decimal places. Instead, use the Oracle Database-specific FLOAT datatype and specify the FLOAT.binary_precision parameter. This is new in version 2.0 of SQLAlchemy.

To create a database agnostic Float that separately specifies binary precision for Oracle Database, use TypeEngine.with_variant() as follows:

from sqlalchemy import Column
from sqlalchemy import Float
from sqlalchemy.dialects import oracle

Column(
    "float_data",
    Float(5).with_variant(oracle.FLOAT(binary_precision=16), "oracle"),
)
asdecimal – the same flag as that of Numeric, but defaults to False. Note that setting this flag to True results in floating point conversion.

decimal_return_scale – Default scale to use when converting from floats to Python decimals. Floating point values will typically be much longer due to decimal inaccuracy, and most floating point database types don’t have a notion of “scale”, so by default the float type looks for the first ten decimal places when converting. Specifying this value will override that length. Note that the MySQL float types, which do include “scale”, will use “scale” as the default for decimal_return_scale, if not otherwise specified.

class sqlalchemy.types.FLOAT
The SQL FLOAT type.

See also

Float - documentation for the base type.

Members

__init__()

Class signature

class sqlalchemy.types.FLOAT (sqlalchemy.types.Float)

method sqlalchemy.types.FLOAT.__init__(precision: int | None = None, asdecimal: bool = False, decimal_return_scale: int | None = None)
inherited from the sqlalchemy.types.Float.__init__ method of Float

Construct a Float.

Parameters:
precision –

the numeric precision for use in DDL CREATE TABLE. Backends should attempt to ensure this precision indicates a number of digits for the generic Float datatype.

Note

For the Oracle Database backend, the Float.precision parameter is not accepted when rendering DDL, as Oracle Database does not support float precision specified as a number of decimal places. Instead, use the Oracle Database-specific FLOAT datatype and specify the FLOAT.binary_precision parameter. This is new in version 2.0 of SQLAlchemy.

To create a database agnostic Float that separately specifies binary precision for Oracle Database, use TypeEngine.with_variant() as follows:

from sqlalchemy import Column
from sqlalchemy import Float
from sqlalchemy.dialects import oracle

Column(
    "float_data",
    Float(5).with_variant(oracle.FLOAT(binary_precision=16), "oracle"),
)
asdecimal – the same flag as that of Numeric, but defaults to False. Note that setting this flag to True results in floating point conversion.

decimal_return_scale – Default scale to use when converting from floats to Python decimals. Floating point values will typically be much longer due to decimal inaccuracy, and most floating point database types don’t have a notion of “scale”, so by default the float type looks for the first ten decimal places when converting. Specifying this value will override that length. Note that the MySQL float types, which do include “scale”, will use “scale” as the default for decimal_return_scale, if not otherwise specified.

attribute sqlalchemy.types..sqlalchemy.types.INT
alias of INTEGER

class sqlalchemy.types.JSON
Represent a SQL JSON type.

Note

JSON is provided as a facade for vendor-specific JSON types. Since it supports JSON SQL operations, it only works on backends that have an actual JSON type, currently:

PostgreSQL - see sqlalchemy.dialects.postgresql.JSON and sqlalchemy.dialects.postgresql.JSONB for backend-specific notes

MySQL - see sqlalchemy.dialects.mysql.JSON for backend-specific notes

SQLite as of version 3.9 - see sqlalchemy.dialects.sqlite.JSON for backend-specific notes

Microsoft SQL Server 2016 and later - see sqlalchemy.dialects.mssql.JSON for backend-specific notes

JSON is part of the Core in support of the growing popularity of native JSON datatypes.

The JSON type stores arbitrary JSON format data, e.g.:

data_table = Table(
    "data_table",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("data", JSON),
)

with engine.connect() as conn:
    conn.execute(
        data_table.insert(), {"data": {"key1": "value1", "key2": "value2"}}
    )
JSON-Specific Expression Operators

The JSON datatype provides these additional SQL operations:

Keyed index operations:

data_table.c.data["some key"]
Integer index operations:

data_table.c.data[3]
Path index operations:

data_table.c.data[("key_1", "key_2", 5, ..., "key_n")]
Data casters for specific JSON element types, subsequent to an index or path operation being invoked:

data_table.c.data["some key"].as_integer()
New in version 1.3.11.

Additional operations may be available from the dialect-specific versions of JSON, such as sqlalchemy.dialects.postgresql.JSON and sqlalchemy.dialects.postgresql.JSONB which both offer additional PostgreSQL-specific operations.

Casting JSON Elements to Other Types

Index operations, i.e. those invoked by calling upon the expression using the Python bracket operator as in some_column['some key'], return an expression object whose type defaults to JSON by default, so that further JSON-oriented instructions may be called upon the result type. However, it is likely more common that an index operation is expected to return a specific scalar element, such as a string or integer. In order to provide access to these elements in a backend-agnostic way, a series of data casters are provided:

Comparator.as_string() - return the element as a string

Comparator.as_boolean() - return the element as a boolean

Comparator.as_float() - return the element as a float

Comparator.as_integer() - return the element as an integer

These data casters are implemented by supporting dialects in order to assure that comparisons to the above types will work as expected, such as:

# integer comparison
data_table.c.data["some_integer_key"].as_integer() == 5

# boolean comparison
data_table.c.data["some_boolean"].as_boolean() == True
New in version 1.3.11: Added type-specific casters for the basic JSON data element types.

Note

The data caster functions are new in version 1.3.11, and supersede the previous documented approaches of using CAST; for reference, this looked like:

from sqlalchemy import cast, type_coerce
from sqlalchemy import String, JSON

cast(data_table.c.data["some_key"], String) == type_coerce(55, JSON)
The above case now works directly as:

data_table.c.data["some_key"].as_integer() == 5
For details on the previous comparison approach within the 1.3.x series, see the documentation for SQLAlchemy 1.2 or the included HTML files in the doc/ directory of the version’s distribution.

Detecting Changes in JSON columns when using the ORM

The JSON type, when used with the SQLAlchemy ORM, does not detect in-place mutations to the structure. In order to detect these, the sqlalchemy.ext.mutable extension must be used, most typically using the MutableDict class. This extension will allow “in-place” changes to the datastructure to produce events which will be detected by the unit of work. See the example at HSTORE for a simple example involving a dictionary.

Alternatively, assigning a JSON structure to an ORM element that replaces the old one will always trigger a change event.

Support for JSON null vs. SQL NULL

When working with NULL values, the JSON type recommends the use of two specific constants in order to differentiate between a column that evaluates to SQL NULL, e.g. no value, vs. the JSON-encoded string of "null". To insert or select against a value that is SQL NULL, use the constant null(). This symbol may be passed as a parameter value specifically when using the JSON datatype, which contains special logic that interprets this symbol to mean that the column value should be SQL NULL as opposed to JSON "null":

from sqlalchemy import null

conn.execute(table.insert(), {"json_value": null()})
To insert or select against a value that is JSON "null", use the constant JSON.NULL:

conn.execute(table.insert(), {"json_value": JSON.NULL})
The JSON type supports a flag JSON.none_as_null which when set to True will result in the Python constant None evaluating to the value of SQL NULL, and when set to False results in the Python constant None evaluating to the value of JSON "null". The Python value None may be used in conjunction with either JSON.NULL and null() in order to indicate NULL values, but care must be taken as to the value of the JSON.none_as_null in these cases.

Customizing the JSON Serializer

The JSON serializer and deserializer used by JSON defaults to Python’s json.dumps and json.loads functions; in the case of the psycopg2 dialect, psycopg2 may be using its own custom loader function.

In order to affect the serializer / deserializer, they are currently configurable at the create_engine() level via the create_engine.json_serializer and create_engine.json_deserializer parameters. For example, to turn off ensure_ascii:

engine = create_engine(
    "sqlite://",
    json_serializer=lambda obj: json.dumps(obj, ensure_ascii=False),
)
Changed in version 1.3.7: SQLite dialect’s json_serializer and json_deserializer parameters renamed from _json_serializer and _json_deserializer.

See also

sqlalchemy.dialects.postgresql.JSON

sqlalchemy.dialects.postgresql.JSONB

sqlalchemy.dialects.mysql.JSON

sqlalchemy.dialects.sqlite.JSON

Members

as_boolean(), as_float(), as_integer(), as_json(), as_numeric(), as_string(), bind_processor(), literal_processor(), NULL, __init__(), bind_processor(), comparator_factory, hashable, python_type, result_processor(), should_evaluate_none

Class signature

class sqlalchemy.types.JSON (sqlalchemy.types.Indexable, sqlalchemy.types.TypeEngine)

class Comparator
Define comparison operations for JSON.

Class signature

class sqlalchemy.types.JSON.Comparator (sqlalchemy.types.Comparator, sqlalchemy.types.Comparator)

method sqlalchemy.types.JSON.Comparator.as_boolean()
Consider an indexed value as boolean.

This is similar to using type_coerce, and will usually not apply a CAST().

e.g.:

stmt = select(mytable.c.json_column["some_data"].as_boolean()).where(
    mytable.c.json_column["some_data"].as_boolean() == True
)
New in version 1.3.11.

method sqlalchemy.types.JSON.Comparator.as_float()
Consider an indexed value as float.

This is similar to using type_coerce, and will usually not apply a CAST().

e.g.:

stmt = select(mytable.c.json_column["some_data"].as_float()).where(
    mytable.c.json_column["some_data"].as_float() == 29.75
)
New in version 1.3.11.

method sqlalchemy.types.JSON.Comparator.as_integer()
Consider an indexed value as integer.

This is similar to using type_coerce, and will usually not apply a CAST().

e.g.:

stmt = select(mytable.c.json_column["some_data"].as_integer()).where(
    mytable.c.json_column["some_data"].as_integer() == 5
)
New in version 1.3.11.

method sqlalchemy.types.JSON.Comparator.as_json()
Consider an indexed value as JSON.

This is similar to using type_coerce, and will usually not apply a CAST().

e.g.:

stmt = select(mytable.c.json_column["some_data"].as_json())
This is typically the default behavior of indexed elements in any case.

Note that comparison of full JSON structures may not be supported by all backends.

New in version 1.3.11.

method sqlalchemy.types.JSON.Comparator.as_numeric(precision, scale, asdecimal=True)
Consider an indexed value as numeric/decimal.

This is similar to using type_coerce, and will usually not apply a CAST().

e.g.:

stmt = select(mytable.c.json_column["some_data"].as_numeric(10, 6)).where(
    mytable.c.json_column["some_data"].as_numeric(10, 6) == 29.75
)
New in version 1.4.0b2.

method sqlalchemy.types.JSON.Comparator.as_string()
Consider an indexed value as string.

This is similar to using type_coerce, and will usually not apply a CAST().

e.g.:

stmt = select(mytable.c.json_column["some_data"].as_string()).where(
    mytable.c.json_column["some_data"].as_string() == "some string"
)
New in version 1.3.11.

class JSONElementType
Common function for index / path elements in a JSON expression.

Class signature

class sqlalchemy.types.JSON.JSONElementType (sqlalchemy.types.TypeEngine)

method sqlalchemy.types.JSON.JSONElementType.bind_processor(dialect: Dialect) → _BindProcessorType[Any]
Return a conversion function for processing bind values.

Returns a callable which will receive a bind parameter value as the sole positional argument and will return a value to send to the DB-API.

If processing is not necessary, the method should return None.

Note

This method is only called relative to a dialect specific type object, which is often private to a dialect in use and is not the same type object as the public facing one, which means it’s not feasible to subclass a TypeEngine class in order to provide an alternate TypeEngine.bind_processor() method, unless subclassing the UserDefinedType class explicitly.

To provide alternate behavior for TypeEngine.bind_processor(), implement a TypeDecorator class and provide an implementation of TypeDecorator.process_bind_param().

See also

Augmenting Existing Types

Parameters:
dialect – Dialect instance in use.

method sqlalchemy.types.JSON.JSONElementType.literal_processor(dialect: Dialect) → _LiteralProcessorType[Any]
Return a conversion function for processing literal values that are to be rendered directly without using binds.

This function is used when the compiler makes use of the “literal_binds” flag, typically used in DDL generation as well as in certain scenarios where backends don’t accept bound parameters.

Returns a callable which will receive a literal Python value as the sole positional argument and will return a string representation to be rendered in a SQL statement.

Note

This method is only called relative to a dialect specific type object, which is often private to a dialect in use and is not the same type object as the public facing one, which means it’s not feasible to subclass a TypeEngine class in order to provide an alternate TypeEngine.literal_processor() method, unless subclassing the UserDefinedType class explicitly.

To provide alternate behavior for TypeEngine.literal_processor(), implement a TypeDecorator class and provide an implementation of TypeDecorator.process_literal_param().

See also

Augmenting Existing Types

class JSONIndexType
Placeholder for the datatype of a JSON index value.

This allows execution-time processing of JSON index values for special syntaxes.

Class signature

class sqlalchemy.types.JSON.JSONIndexType (sqlalchemy.types.JSONElementType)

class JSONIntIndexType
Placeholder for the datatype of a JSON index value.

This allows execution-time processing of JSON index values for special syntaxes.

Class signature

class sqlalchemy.types.JSON.JSONIntIndexType (sqlalchemy.types.JSONIndexType)

class JSONPathType
Placeholder type for JSON path operations.

This allows execution-time processing of a path-based index value into a specific SQL syntax.

Class signature

class sqlalchemy.types.JSON.JSONPathType (sqlalchemy.types.JSONElementType)

class JSONStrIndexType
Placeholder for the datatype of a JSON index value.

This allows execution-time processing of JSON index values for special syntaxes.

Class signature

class sqlalchemy.types.JSON.JSONStrIndexType (sqlalchemy.types.JSONIndexType)

attribute sqlalchemy.types.JSON.NULL = symbol('JSON_NULL')
Describe the json value of NULL.

This value is used to force the JSON value of "null" to be used as the value. A value of Python None will be recognized either as SQL NULL or JSON "null", based on the setting of the JSON.none_as_null flag; the JSON.NULL constant can be used to always resolve to JSON "null" regardless of this setting. This is in contrast to the null() construct, which always resolves to SQL NULL. E.g.:

from sqlalchemy import null
from sqlalchemy.dialects.postgresql import JSON

# will *always* insert SQL NULL
obj1 = MyObject(json_value=null())

# will *always* insert JSON string "null"
obj2 = MyObject(json_value=JSON.NULL)

session.add_all([obj1, obj2])
session.commit()
In order to set JSON NULL as a default value for a column, the most transparent method is to use text():

Table(
    "my_table", metadata, Column("json_data", JSON, default=text("'null'"))
)
While it is possible to use JSON.NULL in this context, the JSON.NULL value will be returned as the value of the column, which in the context of the ORM or other repurposing of the default value, may not be desirable. Using a SQL expression means the value will be re-fetched from the database within the context of retrieving generated defaults.

method sqlalchemy.types.JSON.__init__(none_as_null: bool = False)
Construct a JSON type.

Parameters:
none_as_null=False –

if True, persist the value None as a SQL NULL value, not the JSON encoding of null. Note that when this flag is False, the null() construct can still be used to persist a NULL value, which may be passed directly as a parameter value that is specially interpreted by the JSON type as SQL NULL:

from sqlalchemy import null

conn.execute(table.insert(), {"data": null()})
Note

JSON.none_as_null does not apply to the values passed to Column.default and Column.server_default; a value of None passed for these parameters means “no default present”.

Additionally, when used in SQL comparison expressions, the Python value None continues to refer to SQL null, and not JSON NULL. The JSON.none_as_null flag refers explicitly to the persistence of the value within an INSERT or UPDATE statement. The JSON.NULL value should be used for SQL expressions that wish to compare to JSON null.

See also

JSON.NULL

method sqlalchemy.types.JSON.bind_processor(dialect)
Return a conversion function for processing bind values.

Returns a callable which will receive a bind parameter value as the sole positional argument and will return a value to send to the DB-API.

If processing is not necessary, the method should return None.

Note

This method is only called relative to a dialect specific type object, which is often private to a dialect in use and is not the same type object as the public facing one, which means it’s not feasible to subclass a TypeEngine class in order to provide an alternate TypeEngine.bind_processor() method, unless subclassing the UserDefinedType class explicitly.

To provide alternate behavior for TypeEngine.bind_processor(), implement a TypeDecorator class and provide an implementation of TypeDecorator.process_bind_param().

See also

Augmenting Existing Types

Parameters:
dialect – Dialect instance in use.

attribute sqlalchemy.types.JSON.comparator_factory
alias of Comparator

attribute sqlalchemy.types.JSON.hashable = False
Flag, if False, means values from this type aren’t hashable.

Used by the ORM when uniquing result lists.

attribute sqlalchemy.types.JSON.python_type
method sqlalchemy.types.JSON.result_processor(dialect, coltype)
Return a conversion function for processing result row values.

Returns a callable which will receive a result row column value as the sole positional argument and will return a value to return to the user.

If processing is not necessary, the method should return None.

Note

This method is only called relative to a dialect specific type object, which is often private to a dialect in use and is not the same type object as the public facing one, which means it’s not feasible to subclass a TypeEngine class in order to provide an alternate TypeEngine.result_processor() method, unless subclassing the UserDefinedType class explicitly.

To provide alternate behavior for TypeEngine.result_processor(), implement a TypeDecorator class and provide an implementation of TypeDecorator.process_result_value().

See also

Augmenting Existing Types

Parameters:
dialect – Dialect instance in use.

coltype – DBAPI coltype argument received in cursor.description.

attribute sqlalchemy.types.JSON.should_evaluate_none: bool
If True, the Python constant None is considered to be handled explicitly by this type.

The ORM uses this flag to indicate that a positive value of None is passed to the column in an INSERT statement, rather than omitting the column from the INSERT statement which has the effect of firing off column-level defaults. It also allows types which have special behavior for Python None, such as a JSON type, to indicate that they’d like to handle the None value explicitly.

To set this flag on an existing type, use the TypeEngine.evaluates_none() method.

See also

TypeEngine.evaluates_none()

class sqlalchemy.types.INTEGER
The SQL INT or INTEGER type.

See also

Integer - documentation for the base type.

Class signature

class sqlalchemy.types.INTEGER (sqlalchemy.types.Integer)

class sqlalchemy.types.NCHAR
The SQL NCHAR type.

Members

__init__()

Class signature

class sqlalchemy.types.NCHAR (sqlalchemy.types.Unicode)

method sqlalchemy.types.NCHAR.__init__(length: int | None = None, collation: str | None = None)
inherited from the sqlalchemy.types.String.__init__ method of String

Create a string-holding type.

Parameters:
length – optional, a length for the column for use in DDL and CAST expressions. May be safely omitted if no CREATE TABLE will be issued. Certain databases may require a length for use in DDL, and will raise an exception when the CREATE TABLE DDL is issued if a VARCHAR with no length is included. Whether the value is interpreted as bytes or characters is database specific.

collation –

Optional, a column-level collation for use in DDL and CAST expressions. Renders using the COLLATE keyword supported by SQLite, MySQL, and PostgreSQL. E.g.:

>>> from sqlalchemy import cast, select, String
>>> print(select(cast("some string", String(collation="utf8"))))
SELECT CAST(:param_1 AS VARCHAR COLLATE utf8) AS anon_1
Note

In most cases, the Unicode or UnicodeText datatypes should be used for a Column that expects to store non-ascii data. These datatypes will ensure that the correct types are used on the database.

class sqlalchemy.types.NVARCHAR
The SQL NVARCHAR type.

Members

__init__()

Class signature

class sqlalchemy.types.NVARCHAR (sqlalchemy.types.Unicode)

method sqlalchemy.types.NVARCHAR.__init__(length: int | None = None, collation: str | None = None)
inherited from the sqlalchemy.types.String.__init__ method of String

Create a string-holding type.

Parameters:
length – optional, a length for the column for use in DDL and CAST expressions. May be safely omitted if no CREATE TABLE will be issued. Certain databases may require a length for use in DDL, and will raise an exception when the CREATE TABLE DDL is issued if a VARCHAR with no length is included. Whether the value is interpreted as bytes or characters is database specific.

collation –

Optional, a column-level collation for use in DDL and CAST expressions. Renders using the COLLATE keyword supported by SQLite, MySQL, and PostgreSQL. E.g.:

>>> from sqlalchemy import cast, select, String
>>> print(select(cast("some string", String(collation="utf8"))))
SELECT CAST(:param_1 AS VARCHAR COLLATE utf8) AS anon_1
Note

In most cases, the Unicode or UnicodeText datatypes should be used for a Column that expects to store non-ascii data. These datatypes will ensure that the correct types are used on the database.

class sqlalchemy.types.NUMERIC
The SQL NUMERIC type.

See also

Numeric - documentation for the base type.

Members

__init__()

Class signature

class sqlalchemy.types.NUMERIC (sqlalchemy.types.Numeric)

method sqlalchemy.types.NUMERIC.__init__(precision: int | None = None, scale: int | None = None, decimal_return_scale: int | None = None, asdecimal: bool = True)
inherited from the sqlalchemy.types.Numeric.__init__ method of Numeric

Construct a Numeric.

Parameters:
precision – the numeric precision for use in DDL CREATE TABLE.

scale – the numeric scale for use in DDL CREATE TABLE.

asdecimal – default True. Return whether or not values should be sent as Python Decimal objects, or as floats. Different DBAPIs send one or the other based on datatypes - the Numeric type will ensure that return values are one or the other across DBAPIs consistently.

decimal_return_scale – Default scale to use when converting from floats to Python decimals. Floating point values will typically be much longer due to decimal inaccuracy, and most floating point database types don’t have a notion of “scale”, so by default the float type looks for the first ten decimal places when converting. Specifying this value will override that length. Types which do include an explicit “.scale” value, such as the base Numeric as well as the MySQL float types, will use the value of “.scale” as the default for decimal_return_scale, if not otherwise specified.

When using the Numeric type, care should be taken to ensure that the asdecimal setting is appropriate for the DBAPI in use - when Numeric applies a conversion from Decimal->float or float-> Decimal, this conversion incurs an additional performance overhead for all result columns received.

DBAPIs that return Decimal natively (e.g. psycopg2) will have better accuracy and higher performance with a setting of True, as the native translation to Decimal reduces the amount of floating- point issues at play, and the Numeric type itself doesn’t need to apply any further conversions. However, another DBAPI which returns floats natively will incur an additional conversion overhead, and is still subject to floating point data loss - in which case asdecimal=False will at least remove the extra conversion overhead.

class sqlalchemy.types.REAL
The SQL REAL type.

See also

Float - documentation for the base type.

Members

__init__()

Class signature

class sqlalchemy.types.REAL (sqlalchemy.types.Float)

method sqlalchemy.types.REAL.__init__(precision: int | None = None, asdecimal: bool = False, decimal_return_scale: int | None = None)
inherited from the sqlalchemy.types.Float.__init__ method of Float

Construct a Float.

Parameters:
precision –

the numeric precision for use in DDL CREATE TABLE. Backends should attempt to ensure this precision indicates a number of digits for the generic Float datatype.

Note

For the Oracle Database backend, the Float.precision parameter is not accepted when rendering DDL, as Oracle Database does not support float precision specified as a number of decimal places. Instead, use the Oracle Database-specific FLOAT datatype and specify the FLOAT.binary_precision parameter. This is new in version 2.0 of SQLAlchemy.

To create a database agnostic Float that separately specifies binary precision for Oracle Database, use TypeEngine.with_variant() as follows:

from sqlalchemy import Column
from sqlalchemy import Float
from sqlalchemy.dialects import oracle

Column(
    "float_data",
    Float(5).with_variant(oracle.FLOAT(binary_precision=16), "oracle"),
)
asdecimal – the same flag as that of Numeric, but defaults to False. Note that setting this flag to True results in floating point conversion.

decimal_return_scale – Default scale to use when converting from floats to Python decimals. Floating point values will typically be much longer due to decimal inaccuracy, and most floating point database types don’t have a notion of “scale”, so by default the float type looks for the first ten decimal places when converting. Specifying this value will override that length. Note that the MySQL float types, which do include “scale”, will use “scale” as the default for decimal_return_scale, if not otherwise specified.

class sqlalchemy.types.SMALLINT
The SQL SMALLINT type.

See also

SmallInteger - documentation for the base type.

Class signature

class sqlalchemy.types.SMALLINT (sqlalchemy.types.SmallInteger)

class sqlalchemy.types.TEXT
The SQL TEXT type.

Members

__init__()

Class signature

class sqlalchemy.types.TEXT (sqlalchemy.types.Text)

method sqlalchemy.types.TEXT.__init__(length: int | None = None, collation: str | None = None)
inherited from the sqlalchemy.types.String.__init__ method of String

Create a string-holding type.

Parameters:
length – optional, a length for the column for use in DDL and CAST expressions. May be safely omitted if no CREATE TABLE will be issued. Certain databases may require a length for use in DDL, and will raise an exception when the CREATE TABLE DDL is issued if a VARCHAR with no length is included. Whether the value is interpreted as bytes or characters is database specific.

collation –

Optional, a column-level collation for use in DDL and CAST expressions. Renders using the COLLATE keyword supported by SQLite, MySQL, and PostgreSQL. E.g.:

>>> from sqlalchemy import cast, select, String
>>> print(select(cast("some string", String(collation="utf8"))))
SELECT CAST(:param_1 AS VARCHAR COLLATE utf8) AS anon_1
Note

In most cases, the Unicode or UnicodeText datatypes should be used for a Column that expects to store non-ascii data. These datatypes will ensure that the correct types are used on the database.

class sqlalchemy.types.TIME
The SQL TIME type.

Class signature

class sqlalchemy.types.TIME (sqlalchemy.types.Time)

class sqlalchemy.types.TIMESTAMP
The SQL TIMESTAMP type.

TIMESTAMP datatypes have support for timezone storage on some backends, such as PostgreSQL and Oracle Database. Use the TIMESTAMP.timezone argument in order to enable “TIMESTAMP WITH TIMEZONE” for these backends.

Members

__init__(), get_dbapi_type()

Class signature

class sqlalchemy.types.TIMESTAMP (sqlalchemy.types.DateTime)

method sqlalchemy.types.TIMESTAMP.__init__(timezone: bool = False)
Construct a new TIMESTAMP.

Parameters:
timezone – boolean. Indicates that the TIMESTAMP type should enable timezone support, if available on the target database. On a per-dialect basis is similar to “TIMESTAMP WITH TIMEZONE”. If the target database does not support timezones, this flag is ignored.

method sqlalchemy.types.TIMESTAMP.get_dbapi_type(dbapi)
Return the corresponding type object from the underlying DB-API, if any.

This can be useful for calling setinputsizes(), for example.

class sqlalchemy.types.UUID
Represent the SQL UUID type.

This is the SQL-native form of the Uuid database agnostic datatype, and is backwards compatible with the previous PostgreSQL-only version of UUID.

The UUID datatype only works on databases that have a SQL datatype named UUID. It will not function for backends which don’t have this exact-named type, including SQL Server. For backend-agnostic UUID values with native support, including for SQL Server’s UNIQUEIDENTIFIER datatype, use the Uuid datatype.

New in version 2.0.

See also

Uuid

Members

__init__()

Class signature

class sqlalchemy.types.UUID (sqlalchemy.types.Uuid, sqlalchemy.types.NativeForEmulated)

method sqlalchemy.types.UUID.__init__(as_uuid: bool = True)
Construct a UUID type.

Parameters:
as_uuid=True –

if True, values will be interpreted as Python uuid objects, converting to/from string via the DBAPI.

class sqlalchemy.types.VARBINARY
The SQL VARBINARY type.

Class signature

class sqlalchemy.types.VARBINARY (sqlalchemy.types._Binary)

class sqlalchemy.types.VARCHAR
The SQL VARCHAR type.

Members

__init__()

Class signature

class sqlalchemy.types.VARCHAR (sqlalchemy.types.String)

method sqlalchemy.types.VARCHAR.__init__(length: int | None = None, collation: str | None = None)
inherited from the sqlalchemy.types.String.__init__ method of String

Create a string-holding type.

Parameters:
length – optional, a length for the column for use in DDL and CAST expressions. May be safely omitted if no CREATE TABLE will be issued. Certain databases may require a length for use in DDL, and will raise an exception when the CREATE TABLE DDL is issued if a VARCHAR with no length is included. Whether the value is interpreted as bytes or characters is database specific.

collation –

Optional, a column-level collation for use in DDL and CAST expressions. Renders using the COLLATE keyword supported by SQLite, MySQL, and PostgreSQL. E.g.:

>>> from sqlalchemy import cast, select, String
>>> print(select(cast("some string", String(collation="utf8"))))
SELECT CAST(:param_1 AS VARCHAR COLLATE utf8) AS anon_1
Note

In most cases, the Unicode or UnicodeText datatypes should be used for a Column that expects to store non-ascii data. These datatypes will ensure that the correct types are used on the database.

Previous: SQL Datatype Objects Next: Custom Types
© Copyright 2007-2025, the SQLAlchemy authors and contributors.
flambé! the dragon and The Alchemist image designs created and generously donated by Rotem Yaari.

Created using Sphinx 7.2.6. Documentation last generated: Fri 07 Mar 2025 09:40:25 AM EST