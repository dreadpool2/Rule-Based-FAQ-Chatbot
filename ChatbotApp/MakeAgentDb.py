import MySQLdb as mysq    

#init cursor which points to db
connec = mysq.connect(host = "localhost", user = "sanyog", password = "f20160635.stowe.pyc!")
cursor = connec.cursor()


#initialization of database and table
sql = 'CREATE DATABASE IF NOT EXISTS `Mammoth`'
cursor.execute(sql)
connec.commit()

sql = 'USE `Mammoth`'
cursor.execute(sql)
connec.commit()

sql = 'CREATE TABLE IF NOT EXISTS `AgentHelp` (`Divider` TEXT, `User` TEXT, `Question` TEXT, `AgentAnswer` TEXT, `Status` TEXT)'
cursor.execute(sql)
connec.commit()


